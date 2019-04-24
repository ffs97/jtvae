import sys
import math
import torch
import rdkit
import random
import argparse

from torch.utils.data import DataLoader
from jtvae import JTNNVAE, Vocab, MolTree, MolTreeDataset, JTNNEncoder, MPN, JTMPN


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--tree_latent_size', type=int, default=56)
parser.add_argument('--mol_latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.tree_latent_size,
                args.mol_latent_size, args.depthT, args.depthG)
model.load_state_dict(torch.load(args.model))
model = model.cuda()


smiles = ["CC1CCCCN1CCCC(C2CCCCC2)(C3CCCCC3)O"]
smiles = ["COC1=CC(OC)=CC([C@@H]2C[NH+](CCC(F)(F)F)CC2)=C1"]
print smiles[0]

model.ind = 0
# torch.manual_seed(0)
for i in xrange(args.nsample):
    print model.sample(smiles=smiles, vocab=vocab)
