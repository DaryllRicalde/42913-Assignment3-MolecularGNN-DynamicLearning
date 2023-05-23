# Forked by Group 5 for UTS 42913 Social Information and Network Analysis Assignment 3

Original repository: https://github.com/YohannPerron/MolecularGNN

Our project tackles a future development point mentioned by Pancino et al. (2022)
in their paper on the MolecularGNN model. Our project is an implementation of their proposed
future development which was a  dynamic learning approach that considers a 
dynamic topology of the graphs. The changes between connections are dictated by the 
similarity between neural fingerprints.

We implemented our solution on top of the existing MolecularGNN repository. Our contributions are as follows:

- A custom training function (called "custom_fit") that realizes the idea of dynamic learning
located in `GNN_MinN_classifier.py`
- An implementation of a normalised dot product which was used in calculating neural fingerprint
similarity - see `GNN_node_classifier.py`
- Modified the `load_drugs()` function (see `GNN_node_classifier.py`) so that if it was called during training, the connections
of the drug nodes would be based on whether the normalised dot product similarity of the neural fingerprints are more than
a threshold value. The value used in our report was 0.98. 
- An implementation of neural fingerprint similarity calculation that uses the similarity results
to reset drug-drug connections - see `GNN_node_classifier.py`
- Our group also added graphs for visualising training performance.

<hr>

# MolecularGNN
The MolecularGNN model is a classifier is based on Graph Neural Networks, a connectionist model capable of processing data in form of graphs. This is an improvement over a previous method, called DruGNN[2], as it is also capable of extracting information from the graph--based molecular structures, producing a task--based neural fingerprint (NF) of the molecule which is adapted to the specific task.

# Use
This repository contained 3 training scripts to trained Deep Neural Network classifier on Drug Side Effect Prediction (DSE):

1. GNN_node_classifier.py which train a DrugGNN model as described in [2]. 
2. GNN_molecule_classifier.py which train a DSE classifier base only on our neural figerprint.
3. GNN_MinN_classifier.py which runs the dynamic learning approach proposed by Group 5. 
4. GNN_MinN_classifier.py_original_training which train the full MolecularGNN model using the custom
fit function implemented by Group 5.

All script can take an optionnal command line argument run_id to differentiate training from one another. All parameters related to leraning must be modify inside of the corresponding training script.

# Credit
This work make use of code from:

1. Niccolò Pancino, Pietro Bongini, Franco Scarselli, Monica Bianchini,
  GNNkeras: A Keras-based library for Graph Neural Networks and homogeneous and heterogeneous graph processing,
  SoftwareX, Volume 18, 2022, 101061, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2022.101061. Source code available at https://github.com/NickDrake117/GNNkeras.
2. Bongini, Pietro & Scarselli, Franco & Bianchini, Monica & Dimitri, Giovanna & Pancino, Niccolò & Lio, Pietro. (2022).
  Modular multi-source prediction of drug side-effects with DruGNN.
  IEEE/ACM transactions on computational biology and bioinformatics PP (2022): n. pag.
  Source code available at https://github.com/PietroMSB/DrugSideEffects.

3.Pancino, N., Perron, Y., Bongini, P., & Scarselli, F. (2022). Drug Side Effect Prediction with Deep Learning Molecular Embedding in a Graph-of-Graphs Domain. Mathematics (Basel), 10(23), 4550–. https://doi.org/10.3390/math10234550