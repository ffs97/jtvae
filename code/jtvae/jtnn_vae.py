import copy
import math
import rdkit
import rdkit.Chem as Chem

from mpn import MPN
from jtmpn import JTMPN
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from mol_tree import Vocab, MolTree
from priors import Normal, Discrete, RBM
from nnutils import create_var, flatten_tensor, avg_pool
from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols

import torch
import torch.nn as nn
import torch.nn.functional as F


class JTNNVAE(nn.Module):
    def __init__(self, vocab, hidden_size, tree_latent_size, mol_latent_size, depthT, depthG):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab

        self.jtnn = JTNNEncoder(
            hidden_size, depthT,
            nn.Embedding(vocab.size(), hidden_size)
        )
        self.decoder = JTNNDecoder(
            vocab, hidden_size, tree_latent_size,
            nn.Embedding(vocab.size(), hidden_size)
        )

        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        self.A_assm = nn.Linear(mol_latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        # self.z_tree = Normal(hidden_size, latent_size)
        # self.z_mol = Normal(hidden_size, latent_size)
        # self.z_tree = Discrete(hidden_size, latent_size, 2)
        # self.z_mol = Discrete(hidden_size, latent_size, 2)

        self.dis_z = RBM(hidden_size, tree_latent_size, mol_latent_size)

        self.gas_z_tree = Normal(hidden_size, tree_latent_size)
        self.gas_z_mol = Normal(hidden_size, mol_latent_size)

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph,
                               bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs)  # bilinear
        scores = torch.bmm(
            x_mol_vecs.unsqueeze(1),
            cand_vecs.unsqueeze(-1)
        ).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for mol_tree in mol_batch:
            comp_nodes = [node for node in mol_tree.nodes if len(
                node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(
            neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1)
                    for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands, aroma_score = enum_assemble(
            cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score).cuda()
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
            cand_vecs = self.jtmpn(
                fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            # prevent prob = 0
            probs = F.softmax(scores.view(1, -1), dim=1).squeeze() + 1e-7
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in xrange(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            # father is already attached
            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            has_error = False
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(
                    y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None:
                    has_error = True
                    if i == 0:
                        pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error:
                return cur_mol, cur_mol

        return None, pre_mol

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_mess, tree_vecs, mol_vecs

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        # currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(
            pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        # Important: tree_mess is a matrix, mess_dict is a python dict
        tree_mess = (tree_mess, mess_dict)

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()  # bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx()
                          for atom in cur_mol.GetAtoms()}

        cur_mol, _ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [
        ], pred_root, None, prob_decode, check_aroma=True)
        if cur_mol is None:
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx(): atom.GetIdx()
                              for atom in cur_mol.GetAtoms()}
            cur_mol, pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [
            ], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None:
                cur_mol = pre_mol

        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def forward(self, x_batch, beta, use_gaussian):
        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_mess, x_tree_vecs, x_mol_vecs = self.encode(
            x_jtenc_holder, x_mpn_holder
        )

        x_vecs = torch.cat((x_tree_vecs, x_mol_vecs), dim=1)
        (dis_z_tree_vecs, dis_z_mol_vecs), dis_kl = self.dis_z(x_vecs)

        if use_gaussian:
            gas_z_tree_vecs, gas_tree_kl = self.gas_z_tree(x_tree_vecs)
            gas_z_mol_vecs, gas_mol_kl = self.gas_z_mol(x_mol_vecs)

            z_tree_vecs = dis_z_tree_vecs * gas_z_tree_vecs
            z_mol_vecs = dis_z_mol_vecs * gas_z_mol_vecs

            kl_div = dis_kl + gas_tree_kl + gas_mol_kl

        else:
            z_tree_vecs = dis_z_tree_vecs
            z_mol_vecs = dis_z_mol_vecs

            kl_div = dis_kl

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(
            x_batch, z_tree_vecs
        )
        assm_loss, assm_acc = self.assm(
            x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess
        )

        recon_loss = word_loss + topo_loss + assm_loss

        return recon_loss + beta * kl_div, recon_loss.item(), kl_div.item(), word_acc, topo_acc, assm_acc

    def sample(self, smiles=None, vocab=None, prob_decode=False):
        if smiles is None:
            dis_z_vecs = self.dis_z.sample_feed(1)
            dis_z_mol_vecs, dis_z_tree_vecs = dis_z_vecs

            gas_z_tree_vecs = self.gas_z_tree.sample_feed(1).cuda()
            gas_z_mol_vecs = self.gas_z_mol.sample_feed(1).cuda()

        elif vocab is not None:
            assert(len(smiles) == 1)

            jtenc_holder, mpn_holder = self.tensorize(smiles, vocab)

            _, x_tree_vecs, x_mol_vecs = self.encode(
                jtenc_holder, mpn_holder
            )

            x_vecs = torch.cat((x_tree_vecs, x_mol_vecs), dim=1)

            dis_z_tree_vecs, dis_z_mol_vecs = self.dis_z(x_vecs, sample=False)

            gas_z_tree_vecs = self.gas_z_tree.sample_feed(
                vecs=x_tree_vecs, gamma=10
            ).cuda()
            gas_z_mol_vecs = self.gas_z_mol.sample_feed(
                vecs=x_mol_vecs, gamma=10
            ).cuda()

        else:
            raise AttributeError

        z_tree_vecs = dis_z_tree_vecs * gas_z_tree_vecs
        z_mol_vecs = dis_z_mol_vecs * gas_z_mol_vecs

        return self.decode(z_tree_vecs, z_mol_vecs, prob_decode)

    def reconstruct(self, smiles, prob_decode=False):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        _, tree_vec, mol_vec = self.encode([mol_tree])

        tree_mean = self.T_mean(tree_vec)
        # Following Mueller et al.
        tree_log_var = -torch.abs(self.T_var(tree_vec))
        mol_mean = self.G_mean(mol_vec)
        # Following Mueller et al.
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        epsilon = create_var(torch.randn(1, self.latent_size / 2), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon
        epsilon = create_var(torch.randn(1, self.latent_size / 2), False)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon
        return self.decode(tree_vec, mol_vec, prob_decode)

    @staticmethod
    def tensorize(smiles, vocab):
        def set_batch_nodeID(mol_batch, vocab):
            tot = 0
            for mol_tree in mol_batch:
                for node in mol_tree.nodes:
                    node.idx = tot
                    node.wid = vocab.get_index(node.smiles)
                    tot += 1

        tree_batch = [MolTree(smiles_) for smiles_ in smiles]

        for mol_tree in tree_batch:
            del mol_tree.mol
            for node in mol_tree.nodes:
                del node.mol

        set_batch_nodeID(tree_batch, vocab)
        smiles_batch = [tree.smiles for tree in tree_batch]
        jtenc_holder, _ = JTNNEncoder.tensorize(tree_batch)
        jtenc_holder = jtenc_holder
        mpn_holder = MPN.tensorize(smiles_batch)

        return jtenc_holder, mpn_holder
