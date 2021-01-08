import pandas as pd
import numpy as np
from utils import smiles2adjoin

df = pd.read_csv('data/chem.txt',sep='\t')

atoms_dict = {}

for i,smiles in enumerate(df['CAN_SMILES']):
    atoms,_ = smiles2adjoin(smiles)
    for atom in atoms:
        if atom in atoms_dict.keys():
            atoms_dict[atom]+=1
        else:
            atoms_dict[atom] = 1

print(atoms_dict)
np.save('data/count_results.npy',atoms_dict)

