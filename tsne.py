import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import  BertModel_test,PredictModel
from dataset import Inference_Dataset
import os
import rdkit
import rdkit.Chem as Chem

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

df = pd.read_csv('data/clf/H_HT.txt',sep='\t')
ds = Inference_Dataset(df['SMILES'].tolist()).get_data()

medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
arch = medium
trained_epoch = 10
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']
addH = arch['addH']

dff = d_model * 2
vocab_size = 17
dropout_rate = 0.1

x, adjoin_matrix, smiles, atoms_list = next(iter(ds.take(1)))
seq = tf.cast(tf.math.equal(x, 0), tf.float32)
mask = seq[:, tf.newaxis, tf.newaxis, :]

model = BertModel_test(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)

preds = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)


# pm = PredictModel()
# temp = pm(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
# pm.load_weights('nopretraining_cls\Ames_7.h5')
# pm.encoder.save_weights('nopretraining_cls\Ames_encoder_7.h5')
model.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
# model.encoder.load_weights('nopretraining_cls\Ames_encoder_7.h5')

embedding_list_list = []
smiles_list_list = []
atoms_list_list = []
batch_list = []
for i,(x,adjoin_matrix,smiles,atoms_list) in enumerate(ds):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    _, _, embeddings = model(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
    embedding_list_list.append(embeddings)
    smiles_list_list.append(smiles)
    atoms_list_list.append(atoms_list)
    batch_list.append(i)


smiles_list = []
for i in smiles_list_list:
    i = i.numpy().tolist()
    for ii in i:
        smiles_list.append(''.join([iii.decode() for iii in ii]))


layers = -1
atom_list = []
embeddings_list = []
counts = -1
nums = {}
for i,j in enumerate(atoms_list_list):
    j = j.numpy()
    embedding_batch = embedding_list_list[i][layers].numpy()
    for ii,jj in enumerate(j):
        counts += 1
        embedding_output = embedding_batch[ii,1:,:]
        jj = jj[1:]
        for iii,jjj in enumerate(jj):
            jjj = jjj.decode()
            # if jjj == 'C' or jjj =='N' or jjj == 'O':
            if jjj=='H' or jjj=='':
                continue
            # if counts > 300 and jjj == 'C':
            #     continue
            # if counts > 600 and jjj == 'N':
            #     continue
            # if counts > 500 and jjj == 'O':
            #     continue
            if jjj in nums.keys():
                nums[jjj] += 1
            else:
                nums[jjj] = 1
            atom_list.append(str(counts) + '_' + str(iii) + jjj)
            embeddings_list.append(embedding_output[iii, :])

print(nums)
print(len(embeddings_list))
from sklearn.manifold import TSNE
tsne = TSNE()
print('start to process')
Y = tsne.fit_transform(np.vstack(embeddings_list))
print('Done')


atom_type_list = []
for i,smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    num_atoms = mol.GetNumAtoms()

    for ii in range(num_atoms):
        atom = mol.GetAtomWithIdx(ii)
        # if not (atom.GetSymbol() in ['C', 'N', 'O']):
        #     continue
        # if i > 300 and atom.GetSymbol()=='C':
        #     continue
        # if i > 600 and atom.GetSymbol()=='N':
        #     continue
        # if i > 500 and atom.GetSymbol()=='O':
        #     continue
        neighbors = []
        bonds= []
        atom_type = None
        for neighbor in atom.GetNeighbors():
            neighbor_type = neighbor.GetSymbol()
            bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(),neighbor.GetIdx()).GetBondType()
            neighbors.append(neighbor_type)
            bonds.append(bond_type)
        if atom.GetSymbol() == 'C':
            if rdkit.Chem.rdchem.BondType.DOUBLE in bonds:
                for iii,bond in enumerate(bonds):
                    if bond == rdkit.Chem.rdchem.BondType.DOUBLE:
                        if neighbors[iii] == 'C':
                            atom_type = 3
                        elif neighbors[iii] == 'O':
                            atom_type = 4


            else:
                if 'N' in neighbors:
                    atom_type = 0
                elif 'O' in neighbors:
                    atom_type = 1
                else:
                    atom_type = 2
            if atom.GetIsAromatic():
                atom_type = 5
        if atom.GetSymbol() == 'O':
            if rdkit.Chem.rdchem.BondType.DOUBLE in bonds:
                if neighbors[0] == 'C':
                    atom_type = 6
            else:
                if len(neighbors)==1:
                    atom_type = 7
                if len(neighbors)==2:
                    atom_type = 8
        if atom.GetSymbol() == 'N':
            if rdkit.Chem.rdchem.BondType.DOUBLE in bonds:
                for iii, bond in enumerate(bonds):
                    if bond == rdkit.Chem.rdchem.BondType.DOUBLE:
                        atom_type = 9
            else:
                if len(bonds)==1:
                    atom_type = 10
                elif len(bonds)==2:
                    atom_type = 11
                else:
                    atom_type = 12
        if atom_type is None:
            atom_type = 13
        atom_type_list.append(atom_type)


atom_to_plot = ['C', 'N', 'O','F', 'S', 'Cl', 'P', 'Br','I']
plot_list = []
for aa in atom_to_plot:
    plot_list.append([i.endswith(aa) for i in atom_list])
cmap = [[0.,1.,0.],[0.,0.,1.],[1.,0.,0.],[0.8,0.8,0],[0.7,0.2,0.3],[0,1,1],[0.5,0.6,0.6],[0.6,0.5,1],[0.2,0.2,0.2],[1,1,0]]
for j,plot_atom in enumerate(plot_list):
    plt.scatter(Y[plot_atom,0],Y[plot_atom,1],c=cmap[j],marker='o',s=3)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 设置坐标标签字体大小
plt.xlabel('tSNE-1', fontsize=20)
plt.ylabel('tSNE-2', fontsize=20)
plt.xlim([-70,79])
# for i in range(150):
#     if atom_list[i].startswith('1'):
#         plt.annotate(atom_list[i], xy = (Y[i,0], Y[i,1]), xytext = (Y[i,0]+0.5, Y[i,1]+0.5)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
plt.legend(atom_to_plot,markerscale=12,loc='upper right',labelspacing=1.5)
plt.show()


atom_to_plot = range(14)  #, 'F', 'S', 'Cl', 'P', 'Br','B','I'
plot_list = []
for aa in atom_to_plot:
    plot_list.append([i==aa for i in atom_type_list])

np.random.seed(17)
plt.figure()

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 设置坐标标签字体大小
plt.xlabel('tSNE-1', fontsize=20)
plt.ylabel('tSNE-2', fontsize=20)
plt.xlim([-70,79])


cmap=[[0.2,0.2,1],[0.4,0.9,0.7], [0.7,0.9,0.3],[0.3,0.7,0.5],[0.4,0.5,0.8],[0.,1.,0.],[0.2,1,1],  [0.8,0.5,0.3],[1,0,0],[1,0.2,0.8],[0.7,0.6,1],[1,0.6,0.3], [1,1,0.0],[0,0,0]]
# cmap = [[0.6,0.3,0.6],[0.4,0.6,0.8],[0,1,1],[0.2,0.3,0.8],[0,0.6,0.9],[0.7,0.9,0.1],[0.1,0.8,0.2],[1,1,0],[1,0.5,0.1],
#         [0.9,0.1,0.1],[0.7,0.5,0.3],[0.5,0,0.1],[0.5,0.5,0.5],[0,0,0]]
for j,plot_atom in enumerate(plot_list):
    plt.scatter(Y[plot_atom,0],Y[plot_atom,1],c=cmap[j],marker='o',s=3)

# for i in range(1000):
#         plt.annotate(atom_type_list[i], xy = (Y[i,0], Y[i,1]), xytext = (Y[i,0]+0.5, Y[i,1]+0.5)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
# for i in range(300):
#     atom = atom_list[i]
#     if atom.startswith('8'):
#         plt.annotate(atom_list[i].split('_')[1], xy = (Y[i,0], Y[i,1]), xytext = (Y[i,0], Y[i,1]),fontsize=10) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
# plt.legend(['C-N','C-O','C-C','C=C','C=O','Aromatic C','O=','-OH','-O-','N=','-NH2','-NH1','N','others'],markerscale=12,loc='upper right',labelspacing=1.5)
# plt.show()


keys = ['C3','A','B','C','D','E','F','G','H','I','J']
values = ['8_3C','9_3C','50_5C','50_11C','57_2C','57_8C','71_17C','25_12C','54_11C','60_10C']
for i in range(2000):
    atom = atom_list[i]
    for count,value in enumerate(values):
        if value == atom:
                print(atom)
                plt.annotate(keys[count], xy = (Y[i,0], Y[i,1]), xytext = (Y[i,0], Y[i,1]),fontsize=15) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
plt.legend(['C-N','C-O','C-C','C=C','C=O','Aromatic C','O=','-OH','-O-','N=','-NH2','-NH1','N','others'],markerscale=12,loc='upper right',labelspacing=1.5)
plt.show()









