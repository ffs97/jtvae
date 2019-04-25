import sys
import math
import torch
import rdkit
import random
import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import DataLoader
from jtvae import JTNNVAE, Vocab, MolTree, MolTreeDataset, JTNNEncoder, MPN, JTMPN


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--nsample", type=int, required=True)
parser.add_argument("--vocab", required=True)
parser.add_argument("--model", required=True)

parser.add_argument("--hidden_size", type=int, default=450)
parser.add_argument("--hierarchical", action="store_true", default=False)
parser.add_argument("--tree_latent_size", type=int, default=56)
parser.add_argument("--mol_latent_size", type=int, default=56)
parser.add_argument("--depthT", type=int, default=20)
parser.add_argument("--depthG", type=int, default=3)

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args.hierarchical, args.hidden_size, args.tree_latent_size,
                args.mol_latent_size, args.depthT, args.depthG)
model.load_state_dict(torch.load(args.model))
model = model.cuda()


nei_mols = []

# DISCRETE

np.random.seed(0)
dis_z_vecs = model.dis_z.sample_feed(7)
dis_z_tree_vecs, dis_z_mol_vecs = dis_z_vecs

np.random.seed(1)
gas_z_tree_vecs = model.gas_z_tree.sample_feed(7).cuda() / 4
np.random.seed(2)
gas_z_mol_vecs = model.gas_z_mol.sample_feed(7).cuda() / 4

for i in range(7):
    for j in range(7):
        z_tree_vecs = dis_z_tree_vecs[i:i+1] * gas_z_tree_vecs[j:j+1]
        z_mol_vecs = dis_z_tree_vecs[i:i+1] * gas_z_mol_vecs[j:j+1]

        nei_mols.append(model.decode(
            z_tree_vecs, z_mol_vecs, prob_decode=False
        ))

# NORMAL
# latent_size = model.tree_latent_size + model.mol_latent_size

# np.random.seed(0)
# x = np.random.randn(latent_size)
# x /= np.linalg.norm(x)

# y = np.random.randn(latent_size)
# y -= y.dot(x) * x
# y /= np.linalg.norm(y)

# delta = [
#     [3, 3, 3, 3, 3, 3, 3],
#     [3, 2, 2, 2, 2, 2, 3],
#     [3, 2, 1, 1, 1, 2, 3],
#     [2, 1, 1, 0, 1, 1, 2],
#     [3, 2, 1, 1, 1, 2, 3],
#     [3, 2, 2, 2, 2, 2, 3],
#     [3, 3, 3, 3, 3, 3, 3],
# ]

# smiles = ["COC1=CC(OC)=CC([C@@H]2C[NH+](CCC(F)(F)F)CC2)=C1"]

# jtenc_holder, mpn_holder = model.tensorize(smiles, vocab)

# _, x_tree_vecs, x_mol_vecs = model.encode(
#     jtenc_holder, mpn_holder
# )

# x_vecs = torch.cat((x_tree_vecs, x_mol_vecs), dim=1)

# dis_z_vecs = model.dis_z(x_vecs, sample=False)
# dis_z_tree_vecs, dis_z_mol_vecs = dis_z_vecs

# dis_z_tree_vecs = torch.ones(model.tree_latent_size).cuda() - dis_z_tree_vecs
# dis_z_mol_vecs = torch.ones(model.mol_latent_size).cuda() - dis_z_mol_vecs

# if args.hierarchical:
#     x_tree_vecs = torch.cat((x_tree_vecs, dis_z_tree_vecs), dim=1)
#     x_mol_vecs = torch.cat((x_mol_vecs, dis_z_mol_vecs), dim=1)

# for dx in range(-3, 4):
#     for dy in range(-3, 4):
#         d_tree, d_mol = torch.split(
#             torch.tensor(
#                 (x * dx + y * dy) * delta[dx+3][dy+3]
#             ).type(torch.FloatTensor).cuda(),
#             (model.tree_latent_size, model.mol_latent_size), dim=0
#         )

#         z_tree_vecs = dis_z_tree_vecs * (gas_z_tree_vecs + d_tree)
#         z_mol_vecs = dis_z_mol_vecs * (gas_z_mol_vecs + d_mol)

#         nei_mols.append(model.decode(
#             z_tree_vecs, z_mol_vecs, prob_decode=False
#         ))

nei_mols = [Chem.MolFromSmiles(s) for s in nei_mols]
img = Draw.MolsToGridImage(
    nei_mols, molsPerRow=7,
    subImgSize=(200, 130)
)
img.save("plots.png")
