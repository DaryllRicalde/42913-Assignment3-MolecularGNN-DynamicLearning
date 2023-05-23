#coding=utf-8

import sys
import os
import pickle
import numpy as np
import pandas
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
import pickle
import rdkit
from rdkit import Chem
import matplotlib.pyplot as plt

from GNN.Models.MLP import MLP
from GNN.Models.GNN import GNNgraphBased
from GNN.graph_class import GraphObject
from GNN.Sequencers.GraphSequencers import MultiGraphSequencer


#network parameters
EPOCHS = 100               	#number of training epochs
STATE_DIM = 50				#node state dimension
HIDDEN_UNITS_OUT_NET = 100	#number of hidden units in the output network
LR = 0.001					#learning rate
MAX_ITER = 6				#maximum number of state convergence iterations
DROPOUT_RATE = 0.1			#dropout rate for MLPs
AGGREGATION = "average"     #can either be "average" or "sum"
ACTIVATION = "tanh"			#activation function
SIDE_EFFECT_COUNT = 280		#number of side-effects to take into account 


#gpu parameters
use_gpu = True
target_gpu = "1"

#script parameters
if len(sys.argv) == 1:
	run_id = 'default'
elif len(sys.argv) == 2:
	run_id = sys.argv[1]
else:
	exit(-1)
path_data = "Datasets/Nuovo/Output/Soglia_100/"
path_results = "Results/Nuovo/LinkPredictor/"+run_id+".txt"
atomic_number = { 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Mg':12, 'Al':13, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Fe':26, 'Co':27, 'As':33, 'Br':35, 'I':53, 'Au':79 }
atomic_label = { 1:'H', 3:'Li', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 11:'Na', 12:'Mg', 13:'Al', 15:'P', 16:'S', 17:'Cl', 19:'K' ,20:'Ca', 25:'Mn', 26:'Fe', 27:'Co', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br', 38:'Sr', 39:'Y', 43:'Tc', 47:'Ag', 53:'I', 56:'Ba', 57:'La', 62:'Sm', 64:'Gd', 78:'Pt', 79:'Au', 81:'Tl',88: 'Ra'}
label_translator = {'H':1, 'C':14, 'N':15, 'O':16, 'S':16, 'F':17, 'P':15, 'Cl':17, 'I':17, 'Br':17, 'Ca':2, 'Mg':2, 'K':1, 'Li':1, 'Co':9, 'As':15, 'B':13, 'Al':13, 'Au':11, 'Fe':8, 'Na':1, 'Pt':10, 'Ag':11, 'Zn': 12, 'La':19, 'Sm':19, 'Tc':7, 'Gd':19, 'Sr':2, 'Y':3, 'Se':16, 'Ba':2, 'Tl':13, 'Ge':14, 'Ga':13, 'Mn':7, 'Ra':2,}
chromosome_dict = {'MT':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20, '21':21, '22':22, 'X':23, 'Y':24}
n_atom_categories = max(label_translator.values())+1

#Data parameters
LABEL_DIM = n_atom_categories + 5
ARC_DIM = 5

#function that adjusts NaN features
def CheckedFeature(feature):
	if feature is None:
		return 0.0
	if np.isnan(feature):
		return 0.0
	return feature

#function that binarizes a fingerprint
def BinarizedFingerprint(fp):
	bfp = [0 for i in range(len(fp))]
	for i in range(len(fp)):
		if fp[i] > 0:
			bfp[i] = 1
	return bfp


def load_molecules():
	#load side-effect data
	print("Loading side-effects")
	in_file = open(path_data+"side_effects.pkl", 'rb')
	side_effects = pickle.load(in_file)
	in_file.close()

	#load drug data
	print("Loading drugs")
	in_file = open(path_data+"drugs.pkl", 'rb')
	drugs_pre = pickle.load(in_file)
	in_file.close()
	#load drug-side effect links
	print("Loading drug - side-effects associations")
	in_file = open(path_data+"drug_side_effect_links.pkl", 'rb')
	links_dse = pickle.load(in_file)
	in_file.close()
	#load drug features
	print("Loading drug features")
	pubchem_data = pandas.read_csv(path_data+"pubchem_output.csv")

	#preprocess drug ids
	print("Preprocessing drug identifiers")
	drugs = list()
	for i_mol in range(len(drugs_pre)):
		drugs.append(str(int(drugs_pre[i_mol][4:])))

	#determine dataset dimensions
	print("Calculating dataset dimensions")
	CLASSES = len(side_effects)		#number of outputs
	n_drugs = len(drugs)
	print("Number of side effects: ", CLASSES)
	print("Number of drugs: ", n_drugs)

	#build id -> node number mappings
	drug_number = dict()
	for i_mol in range(len(drugs)):
		drug_number[str(drugs[i_mol])] = i_mol
	#build id -> class number mappings
	class_number = dict()
	for i_mol in range(len(side_effects)):
		class_number[side_effects[i_mol]] = i_mol

	# #build output mask
	# print("Building output mask")
	# output_mask = np.concatenate((np.ones(len(drugs)), np.zeros(len(genes))))

	#build dict of molecular structures
	print("Building molecular structures")
	molecule_dict = dict()
	for i_mol in pubchem_data.index:
		#skip drugs which were filtered out of the dataset
		if str(pubchem_data.at[i_mol,'cid']) not in drug_number.keys():
			continue
		nn = drug_number[str(pubchem_data.at[i_mol,'cid'])]
		molecule_dict[nn] = rdkit.Chem.MolFromSmiles(pubchem_data.at[i_mol,'isosmiles'])

	#building atomes features (only explicit H and heavy atoms)
	atom_features_dict = dict()
	for i_mol in molecule_dict.keys():
		n_atom = molecule_dict[i_mol].GetNumAtoms()
		features = np.zeros((n_atom,LABEL_DIM))
		for i_atom in range(n_atom):
			atom = molecule_dict[i_mol].GetAtomWithIdx(i_atom)
			Z = atom.GetAtomicNum()
			#onehot encoded atomic number
			features[i_atom,label_translator[atomic_label[Z]]] = 1
			#other features
			features[i_atom,n_atom_categories+0] = atom.GetDegree()
			features[i_atom,n_atom_categories+1] = atom.GetTotalNumHs()
			features[i_atom,n_atom_categories+2] = atom.GetNumImplicitHs()
			features[i_atom,n_atom_categories+3] = atom.GetNumRadicalElectrons()
			features[i_atom,n_atom_categories+4] = atom.GetFormalCharge()
		atom_features_dict[i_mol] = features

	#building arc - links between atoms
	arc_dict = dict()
	for i_mol in molecule_dict.keys():
		n_bond = molecule_dict[i_mol].GetNumBonds(onlyHeavy=True)
		arc = np.zeros((2*n_bond,2+ARC_DIM))
		for i_bond in range(n_bond):
			bond = molecule_dict[i_mol].GetBondWithIdx(i_bond)
			i_start = bond.GetBeginAtomIdx()
			i_end = bond.GetEndAtomIdx()
			bond_type = int((bond.GetBondTypeAsDouble()-1)*2)
			arc[2*i_bond,:2]   = [i_start,i_end]
			arc[2*i_bond,bond_type+2]=1
			arc[2*i_bond+1,:2] = [i_end,i_start]
			arc[2*i_bond+1,bond_type+2]=1
		arc_dict[i_mol]=arc

	#build list of positive examples
	print("Building list of positive examples")
	positive_dsa_list = list()
	for i_mol in range(links_dse.shape[0]):
		if str(int(links_dse[i_mol][0][4:])) in drug_number.keys():
			#skip side-effects which were filtered out of the dataset
			if links_dse[i_mol][2] not in side_effects:
				continue
			positive_dsa_list.append((drug_number[str(int(links_dse[i_mol][0][4:]))],class_number[links_dse[i_mol][2]]))
		else:
			sys.exit("ERROR: drug-side-effect link pointing to incorrect drug id")



	#build target tensor
	print("Building target tensor")
	targets = np.zeros((len(drugs),len(side_effects)))
	for p in positive_dsa_list:	
		targets[p[0]][p[1]] = 1

	#select most common side effects
	se_counts = np.sum(targets, axis=0)
	sorted_counts = np.sort(-se_counts)
	se_count_threshold = -sorted_counts[SIDE_EFFECT_COUNT-1]
	del_indices = list()
	for i_mol in range(targets.shape[1]):
		if se_counts[i_mol] < se_count_threshold:
			del_indices.append(i_mol)
	for i_mol in range(targets.shape[1]):
		if se_counts[i_mol] == se_count_threshold and len(del_indices) < CLASSES - SIDE_EFFECT_COUNT:
			del_indices.append(i_mol)
	targets = np.delete(targets, del_indices, axis=1)
	CLASSES = SIDE_EFFECT_COUNT
	if targets.shape[1] != CLASSES:
		sys.exit("ERROR: Error while selecting most common side-effect")

	### DEBUG START ###
	### DEBUG: calculate graph diameter
	'''
	print("Calculating graph diameter")
	import networkx as nx
	g = nx.Graph()
	for i in range(len(nodes)):
		g.add_node(i)
	for i in range(len(arcs)):
		g.add_edge(arcs[i][0], arcs[i][1])
	diameter = nx.algorithms.distance_measures.diameter(g)
	print("Graph Diameter = "+str(diameter))
	sys.exit()
	'''
	### DEBUG STOP ###

	print("Building GraphObject")
	graphs = list()
	for i_mol in molecule_dict.keys():
		# if arc_dict[i_mol].shape[0]==0:
		# 	print(Chem.MolToSmiles(molecule_dict[i_mol]))
		graphs.append(
			GraphObject(atom_features_dict[i_mol], arc_dict[i_mol], targets[i_mol,:].reshape((1,-1)),focus = 'g',aggregation_mode=AGGREGATION)
			)
	
	return graphs

if __name__ == "__main__":
	#set target gpu as the only visible device
	if use_gpu:
		os.environ["CUDA_VISIBLE_DEVICES"]=target_gpu

	graphs = np.array(load_molecules())

	index = np.array(list(range(len(graphs))))
	np.random.seed(3)
	np.random.shuffle(index)
	test_index = np.sort(index[:133])
	validation_index = np.sort(index[133:266])
	training_index = np.sort(index[266:])
	print(test_index)
	gTr = graphs[training_index].tolist()
	gVa = graphs[validation_index].tolist()
	gTe = graphs[test_index].tolist()
	CLASSES = gTr[0].targets.shape[1]

	gTr_Sequencer = MultiGraphSequencer(gTr, 'g', AGGREGATION, 9999, shuffle=False)
	gVa_Sequencer = MultiGraphSequencer(gVa, 'g', AGGREGATION, 32, shuffle=False)
	gTe_Sequencer = MultiGraphSequencer(gTe, 'g', AGGREGATION, 32, shuffle=False)

	netSt = MLP(input_dim=(2*STATE_DIM+2*LABEL_DIM+ARC_DIM,), layers=[STATE_DIM], activations=[ACTIVATION],
						kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
						bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(1)],
						kernel_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(1)],
						bias_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(1)],
						dropout_rate=DROPOUT_RATE, dropout_pos=1,batch_normalization=False)

	netOut = MLP(input_dim=(STATE_DIM+LABEL_DIM,), layers=[HIDDEN_UNITS_OUT_NET,CLASSES], activations=[ACTIVATION, 'sigmoid'],
						kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
						bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
						kernel_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(2)],
						bias_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(2)],
						dropout_rate=DROPOUT_RATE, dropout_pos=1)

	gnn = GNNgraphBased(netSt, netOut, STATE_DIM, MAX_ITER, 0.01).copy()
	gnn.compile(optimizer=tf.keras.optimizers.Adam(LR),
				loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=5),
				average_st_grads=False,
				metrics=[tf.keras.metrics.BinaryAccuracy(),
						F1Score(num_classes=SIDE_EFFECT_COUNT, threshold=0.5, average='micro')],
				run_eagerly=True)

	gnn.fit(gTr_Sequencer, epochs=EPOCHS, validation_data=gVa_Sequencer)

	# np.save("VA_out_new",gnn.predict(gVa_Sequencer))
	# np.save("TE_out_new",gnn.predict(gTe_Sequencer))
	# np.save("VA_targets_new",gVa_Sequencer.targets)
	# np.save("TE_targets_new",gTe_Sequencer.targets)