#coding=utf-8


import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from GNN.Models.MLP import MLP
from GNN.Models.GNN import GNNgraphBased
from GNN.Models.CompositeGNN import CompositeGNNnodeBased
from GNN.graph_class import GraphObject
from GNN.Sequencers.GraphSequencers import MultiGraphSequencer, CompositeMultiGraphSequencer, CompositeSingleGraphSequencer

from GNN_molecule_classifier import load_molecules, ARC_DIM, LABEL_DIM
from GNN_node_classifier import load_drugs, LABEL_DIMS
from GNN_MinN_utils import MinN_model, MinN_Sequence, weight
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

name="baseline"
#network parameters
EPOCHS = 10                  #number of training epochs
LR = 0.001					#learning rate
AGGREGATION = "average"     #can either be "average" or "sum"
ACTIVATION = "tanh"			#activation function
SIDE_EFFECT_COUNT = 280		#number of side-effects to take into account
INNER_DIM = 380

#gpu parameters
use_gpu = True
target_gpu = "1"

def create_model():
    #molecules embedding submodel
    M_nb_layers = 3
    M_nb_layers_out = 2
    M_batch_norm = True
    M_L2 = 0.001
    M_dropout = 0.0
    M_activation = 'relu'
    M_state_dim = 150
    M_max_iter = 4

    # Net state => Sequential model
    M_netSt = MLP(input_dim=(2*M_state_dim+2*LABEL_DIM+ARC_DIM,),
                        layers=[M_state_dim for i in range(M_nb_layers)],
                        activations=[M_activation for i in range(M_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(M_L2) for i in range(M_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(M_L2) for i in range(M_nb_layers)],
                        dropout_rate=M_dropout, dropout_pos=1,batch_normalization=M_batch_norm)

    # Net output => Sequential model
    M_netOut = MLP(input_dim=(M_state_dim+LABEL_DIM,),
                        layers=[INNER_DIM for i in range(M_nb_layers_out)],
                        activations=[M_activation for i in range(M_nb_layers_out)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers_out)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers_out)],
                        kernel_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(M_nb_layers_out)],
                        bias_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(M_nb_layers_out)],
                        dropout_rate=M_dropout, dropout_pos=1,batch_normalization=M_batch_norm)

    # Create molecular embedding submodel GNN
    moleculesGNN = GNNgraphBased(M_netSt, M_netOut, M_state_dim, M_max_iter, 0.01)

    #drug-effect submodel
    N_nb_layers = 2
    N_batch_norm = True
    N_L2 = 0.0
    N_dropout = 0.1
    N_activation = 'relu'
    N_state_dim =50
    N_max_iter = 4
    HIDDEN_UNITS_OUT_NET = 200

    netSt_drugs = MLP(input_dim=(2*N_state_dim+LABEL_DIMS[0]+sum(LABEL_DIMS),),
                        layers=[N_state_dim for i in range(N_nb_layers)],
                        activations=[N_activation for i in range(N_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)
    netSt_drugs_augmented = MLP(input_dim=(2*N_state_dim+LABEL_DIMS[2]+sum(LABEL_DIMS),), 
                        layers=[N_state_dim for i in range(N_nb_layers)],
                        activations=[N_activation for i in range(N_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)
    netSt_genes = MLP(input_dim=(2*N_state_dim+LABEL_DIMS[1]+sum(LABEL_DIMS),),
                        layers=[N_state_dim for i in range(N_nb_layers)],
                        activations=[N_activation for i in range(N_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(N_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(N_nb_layers)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)
    netOut = MLP(input_dim=(N_state_dim+LABEL_DIMS[0],), layers=[HIDDEN_UNITS_OUT_NET,CLASSES], activations=[N_activation, 'sigmoid'],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(2)],
                        kernel_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(2)],
                        bias_regularizer=[tf.keras.regularizers.L2(N_L2) for i in range(2)],
                        dropout_rate=N_dropout, dropout_pos=1,batch_normalization=N_batch_norm)

    dseGNN = CompositeGNNnodeBased([netSt_drugs, netSt_genes, netSt_drugs_augmented], netOut, N_state_dim, N_max_iter, 0.001)

    gamma = 5
    mu = 0.5

    #calculate class weight
    class_weight = weight(DG_Tr_Sequencer.targets, mu=0.0)

    #define loss function with weight
    BC_object = tf.keras.losses.BinaryFocalCrossentropy(gamma = gamma, reduction = 'none')
    def loss(y_true, y_pred):
        l = BC_object(y_true[..., tf.newaxis], y_pred[..., tf.newaxis])
        l = tf.math.multiply(l, class_weight)
        return tf.reduce_mean(l)

    #create model
    Model = MinN_model(moleculesGNN, dseGNN, embedding_start=LABEL_DIMS[0]-INNER_DIM,embedding_size=INNER_DIM)
    print("Embedding start: ", LABEL_DIMS[0]-INNER_DIM)
    print("Embedding end/size?:", INNER_DIM)
    Model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                loss=loss,
                average_st_grads=False,
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                F1Score(num_classes=SIDE_EFFECT_COUNT, threshold=0.5, average='micro'),
                tf.keras.metrics.AUC(num_thresholds=200,curve='ROC',multi_label=True),
                tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),
                tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.PrecisionAtRecall(0.9)],
                run_eagerly=True)
    return Model


def create_molecule_embedding_submodel_only():
    #molecules embedding submodel
    M_nb_layers = 3
    M_nb_layers_out = 2
    M_batch_norm = True
    M_L2 = 0.001
    M_dropout = 0.0
    M_activation = 'relu'
    M_state_dim = 150
    M_max_iter = 4

    # Net state => Sequential model
    M_netSt = MLP(input_dim=(2*M_state_dim+2*LABEL_DIM+ARC_DIM,),
                        layers=[M_state_dim for i in range(M_nb_layers)],
                        activations=[M_activation for i in range(M_nb_layers)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers)],
                        kernel_regularizer=[tf.keras.regularizers.L2(M_L2) for i in range(M_nb_layers)],
                        bias_regularizer=[tf.keras.regularizers.L2(M_L2) for i in range(M_nb_layers)],
                        dropout_rate=M_dropout, dropout_pos=1,batch_normalization=M_batch_norm)

    # Net output => Sequential model
    M_netOut = MLP(input_dim=(M_state_dim+LABEL_DIM,),
                        layers=[INNER_DIM for i in range(M_nb_layers_out)],
                        activations=[M_activation for i in range(M_nb_layers_out)],
                        kernel_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers_out)],
                        bias_initializer=[tf.keras.initializers.GlorotNormal() for i in range(M_nb_layers_out)],
                        kernel_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(M_nb_layers_out)],
                        bias_regularizer=[tf.keras.regularizers.L2(0.0) for i in range(M_nb_layers_out)],
                        dropout_rate=M_dropout, dropout_pos=1,batch_normalization=M_batch_norm)

    # Create molecular embedding submodel GNN
    moleculesGNN = GNNgraphBased(M_netSt, M_netOut, M_state_dim, M_max_iter, 0.01)
    print("moleculesGNN\n",moleculesGNN.summary())

    gamma = 5
    mu = 0.5

    #calculate class weight
    class_weight = weight(DG_Tr_Sequencer.targets, mu=0.0)

    #define loss function with weight
    BC_object = tf.keras.losses.BinaryFocalCrossentropy(gamma = gamma, reduction = 'none')
    def loss(y_true, y_pred):
        l = BC_object(y_true[..., tf.newaxis], y_pred[..., tf.newaxis])
        l = tf.math.multiply(l, class_weight)
        return tf.reduce_mean(l)

    #create model
    moleculesGNN.compile(optimizer=tf.keras.optimizers.Adam(LR),
                loss=loss,
                average_st_grads=False,
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                F1Score(num_classes=SIDE_EFFECT_COUNT, threshold=0.5, average='micro'),
                tf.keras.metrics.AUC(num_thresholds=200,curve='ROC',multi_label=True),
                tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),
                tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.PrecisionAtRecall(0.9)],
                run_eagerly=True)
    print("moleculesGNN summary\n",moleculesGNN.summary())
    print("----------------------------------")
    return moleculesGNN


def call(self, inputs, training = False):
    #calculate molecule embedding
    molecules, DSE_graph = inputs
    print("Type of DSE_graph", type(DSE_graph))

    # Getting the Neural Fingerprints from the molecular embedding submodel
    if training:
        molecules_embedding = self.moleculeGNN(molecules, training=training)[2]
    else:
        molecules_embedding = self.moleculeGNN(molecules, training=training)

    print("###### Molecules embedding shape #### ", np.shape(molecules_embedding.numpy()))
    print("Molecules embedding: ", molecules_embedding)


    #insert them in the drug node features
    nodes_features = DSE_graph[0]
    type_mask = DSE_graph[3]

    is_drug = tf.reshape(tf.math.logical_or(type_mask[0],type_mask[2]), [-1])
    is_drug_idx = tf.where(is_drug)

    drug_nodes_features = tf.gather(nodes_features, tf.reshape(is_drug_idx, [-1]))
    print("Shape of drug_nodes_features", np.shape(drug_nodes_features.numpy()))

    # Placing the molecules embedding in a particular position
    new_drug_nodes_features = tf.concat([drug_nodes_features[:,:self.embedding_start],molecules_embedding,drug_nodes_features[:,self.embedding_end:]],axis=1)
    print("Shape of new drug_nodes_features", np.shape(new_drug_nodes_features.numpy()))


    new_feature = tf.tensor_scatter_nd_update(DSE_graph[0], is_drug_idx, new_drug_nodes_features)
    # print(" ####### new_feature (maybe neural fingerprint) ####### ", new_feature.numpy())


    new_DSE_graph = (new_feature,)+DSE_graph[1:]

    if training:
        return self.drugGNN(new_DSE_graph, training=training)[2]
    return self.drugGNN(new_DSE_graph, training=training)

#molecules embedding graphs and submodel
graphs = load_molecules()

MG_Sequencer = MultiGraphSequencer(graphs, 'g', AGGREGATION, 32000, shuffle = False)

#drug-effect graphs
DG_Trs, DG_Va, DG_Te = load_drugs(num_batch = 10,molecular_feature_size = INNER_DIM)
print("First graph creation:",DG_Trs)
CLASSES = DG_Te.targets.shape[1]

print("A")
DG_Tr_Sequencer = CompositeMultiGraphSequencer(DG_Trs, 'n', AGGREGATION, 1, shuffle=False)
DG_Va_Sequencer = CompositeSingleGraphSequencer(DG_Va, 'n', AGGREGATION, 9999) 
DG_Te_Sequencer = CompositeSingleGraphSequencer(DG_Te, 'n', AGGREGATION, 9999)

#create data sequences
Tr_sequencer = MinN_Sequence(MG_Sequencer,DG_Tr_Sequencer)
Va_sequencer = MinN_Sequence(MG_Sequencer,DG_Va_Sequencer)
Te_sequencer = MinN_Sequence(MG_Sequencer,DG_Te_Sequencer)


Model = create_model()

#create callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='default/', histogram_freq=1)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience=50, restore_best_weights=True)

training = True

epoch_num_list = []
accuracy_tr_list = [] 
accuracy_val_list = []
loss_tr_list = []
loss_val_list = []

"""
Social Information and Network Analysis Group 5 Contribution: custom_fit(...) function

This function encapsulates the dynamic learning approach we designed for for 42913 Social  Information and Network Analysis
"""

def custom_fit(model, train_dataset , epochs, validation_data, callbacks, DG_Trs, DG_Tr_Sequencer = None):
    model.compile()
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        epoch_tra_accuracy = []
        epoch_val_accuracy = []
        new_DSE_graph_temp_epoch = None
        loss_value_for_epoch = 0

        for i in range(10):
            print("batch num: ", i)
            inputs = tuple(train_dataset[i][0])
        
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer. The operations that the layer applies
                # to its inputs are going to be recorded on the GradientTape.
                print("input size:")
                print(len(inputs))
                #calculate molecule embedding
                molecules, DSE_graph = inputs

                print("Type of DSE_graph", type(DSE_graph[1:]))
                # print(DSE_graph[1:])

                # Getting the Neural Fingerprints from the molecular embedding submodel
                # if training:
                molecules_embedding = Model.moleculeGNN(molecules, training = training)[2]
                # else:
                #     molecules_embedding = self.moleculeGNN(molecules, training=training)
                
                print("###### Molecules embedding shape #### ", np.shape(molecules_embedding.numpy()))
                # print("Molecules embedding: ", molecules_embedding)


                #insert them in the drug node features
                nodes_features = DSE_graph[0]
                type_mask = DSE_graph[3]

                is_drug = tf.reshape(tf.math.logical_or(type_mask[0],type_mask[2]), [-1])
                is_drug_idx = tf.where(is_drug)

                drug_nodes_features = tf.gather(nodes_features, tf.reshape(is_drug_idx, [-1]))
                print("Shape of drug_nodes_features", np.shape(drug_nodes_features.numpy()))

                # Placing the molecules embedding in a particular position 
                new_drug_nodes_features = tf.concat([drug_nodes_features[:,:Model.embedding_start],molecules_embedding,drug_nodes_features[:,Model.embedding_end:]],axis=1)
                print("Shape of new drug_nodes_features", np.shape(new_drug_nodes_features.numpy()))


                new_feature = tf.tensor_scatter_nd_update(DSE_graph[0], is_drug_idx, new_drug_nodes_features)
                # print(" ####### new_feature (maybe neural fingerprint) ####### ", new_feature.numpy())

                
                # new_DSE_graph = (new_feature,)+ DSE_graph[1:]
                new_DSE_graph = (new_feature,)+ tuple(DSE_graph[1:])
                new_DSE_graph_temp_epoch = new_DSE_graph

                # if training:
                model_output = model.drugGNN(new_DSE_graph, training=training)[2]
                # else:
                #     model_output = self.drugGNN(new_DSE_graph, training=training)

                logits = model_output
                # logits = model(Tr_sequencer, training=True)  # Logits for this minibatch
                gamma = 5
                mu = 0.5

                #calculate class weight
                class_weight = weight(DG_Tr_Sequencer.targets, mu=0.0)

                #define loss function with weight
                BC_object = tf.keras.losses.BinaryFocalCrossentropy(gamma = gamma, reduction = 'none')
                def loss(y_true, y_pred):
                    l = BC_object(y_true[..., tf.newaxis], y_pred[..., tf.newaxis])
                    l = tf.math.multiply(l, class_weight)
                    return tf.reduce_mean(l)

                # Compute the loss value for this minibatch.
                loss_value = loss(Tr_sequencer[i][1], logits)
                loss_value_for_epoch += loss_value  

                # Tr_sequencer[i][1].shape == logits.shape == (108,280)
                metric = tf.keras.metrics.BinaryAccuracy()
                metric.update_state(Tr_sequencer[i][1], logits)
                print("Prediction accuracy:", metric.result().numpy())

                # Append prediction accuracy for this iteration to epoch_accuracy
                epoch_tra_accuracy.append(metric.result().numpy())

                # Tr_sequencer[i][1].shape == logits.shape == (108,280)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)
            
            optimizer=tf.keras.optimizers.Adam(LR)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        
        train_loss = loss_value_for_epoch/10 #epoch loss should be divided by the number of graphs in validation data
        
        #get the validation loss and accuracy
        logits_val = model.predict(validation_data)
        metric_val = tf.keras.metrics.BinaryAccuracy()
        metric_val.update_state(validation_data[0][1], logits_val)

        val_loss = model.evaluate(validation_data)

        """
        Training step:
            Get molecules and DSE graph
            Get the NF's from model.moleculeGNN by providing molecules as input
                molecules_embedding = Model.moleculeGNN(molecules, training = training)[2]
            Concatenate new drug node features to DSE Graph, making a new_DSE_graph 
            Get model_output/logits by providing new_DSE_graph to model.druGNN()
                model_output = model.drugGNN(new_DSE_graph, training=training)[2]
        """
            
        # Get epoch's average accuracy
        arr_tra = np.array(epoch_tra_accuracy)
        average_tra = np.mean(arr_tra)
        val_acc = metric_val.result().numpy()
        
        print("Training Accuracy: ", average_tra)
        print("Training Loss: ", train_loss)
        print("Validation Accuracy: ", val_acc)
        print("Validation_loss: ", val_loss)

        #Append loss and accuracy to global lists for displaying the results
        epoch_num_list.append(epoch)
        accuracy_tr_list.append(average_tra)
        accuracy_val_list.append(val_acc)
        loss_tr_list.append(train_loss)
        loss_val_list.append(val_loss)

        #update the drug drug connections by recalling load_drugs and reconstructing the network using the refined neural fingerprints
        DG_Trs, DG_Va_curr_epoch, DG_Va_curr_epoch = load_drugs(num_batch = 10,molecular_feature_size = INNER_DIM, training = True, molecules_embedding = molecules_embedding)
        print("DG_Trs TYPE: ", type(DG_Trs))
        print(DG_Trs)
        #recreate the DG_TR_Sequencer
        DG_Tr_Sequencer = CompositeMultiGraphSequencer(DG_Trs, 'n', AGGREGATION,  1)
        #Reset the train dataset
        train_dataset = MinN_Sequence(MG_Sequencer,DG_Tr_Sequencer)

    return model


#call the new custom fit method to run the dynamic learning approach
Model = custom_fit(Model,Tr_sequencer, EPOCHS, Va_sequencer, callbacks=[tensorboard_callback,earlystop], DG_Trs = DG_Trs, DG_Tr_Sequencer = DG_Tr_Sequencer)

#evaluate
print("Te sequencer shape: ",np.shape(Te_sequencer))
Test_loss = Model.evaluate(Te_sequencer)

logits_test = Model.predict(Te_sequencer)
metric_test = tf.keras.metrics.BinaryAccuracy()
metric_test.update_state(Te_sequencer[0][1], logits_test)

print("Test Accuracy: ", metric_test.result().numpy())
print("Test Loss: ", Test_loss)



# Group 5 Contribution: graphs for visualisation of results

graph_name = "Dynamic_learning_epochs_10"


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(5,8))
plt.suptitle(graph_name, x=0.5, ha="center")

ax[0].plot(accuracy_tr_list, color='orange')
ax[0].set_xlim(left=0)
ax[0].set_ylim(bottom=0, top=1)
ax[0].set_yticks(np.arange(0, 1.1, 0.1))
ax[0].set_yticklabels(['{:.2f}'.format(tick) for tick in ax[0].get_yticks()])
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Train Accuracy")
ax[0].grid(True, linestyle=":", linewidth=0.5)

ax[1].plot(loss_tr_list)
ax[1].set_xlim(left=0-1)
ax[1].set_ylim(bottom=0, top=max(1, max(loss_tr_list)))
ax[1].set_yticks(np.arange(0, max(1, max(loss_tr_list)), 0.2))
ax[1].set_yticklabels(['{:.2f}'.format(tick) for tick in ax[1].get_yticks()])
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Train Loss")
ax[1].grid(True, linestyle=":", linewidth=0.5)

ax[2].plot(accuracy_val_list, color='orange')
ax[2].set_xlim(left=0)
ax[2].set_ylim(bottom=0, top=1)
ax[2].set_yticks(np.arange(0, 1.1, 0.1))
ax[2].set_yticklabels(['{:.2f}'.format(tick) for tick in ax[2].get_yticks()])
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("Validation Accuracy")
ax[2].grid(True, linestyle=":", linewidth=0.5)

ax[3].plot(loss_val_list)
ax[3].set_xlim(left=0-1)
ax[3].set_ylim(bottom=0, top=max(1, max(loss_val_list)))
ax[3].set_yticks(np.arange(0, max(1, max(loss_val_list)), 0.2))
ax[3].set_yticklabels(['{:.2f}'.format(tick) for tick in ax[3].get_yticks()])
ax[3].set_xlabel("Epochs")
ax[3].set_ylabel("Validation Loss")
ax[3].grid(True, linestyle=":", linewidth=0.5)


plt.tight_layout()


plt.savefig('Graphs/' + graph_name + '.png')  # Specify the desired file name and extension

plt.show()
# Close the plot
plt.close()