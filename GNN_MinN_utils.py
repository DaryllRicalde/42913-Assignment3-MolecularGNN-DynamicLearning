#coding=utf-8

from locale import normalize
from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf
import pandas


# tf.keras.utils.Sequence -> Base object for fitting to a sequence of data, such as a dataset
class MinN_Sequence(tf.keras.utils.Sequence):

    def __init__(self, M_Sequencer,D_Sequencer):
        self.M_Sequencer = M_Sequencer
        self.D_Sequencer = D_Sequencer
        self.batch_size = D_Sequencer.batch_size

    def __len__(self):
        return self.D_Sequencer.__len__()

    def __getitem__(self, idx):
        M_batch_x, M_batch_y, M_batch_w = self.M_Sequencer.__getitem__(0)
        D_batch_x, D_batch_y, D_batch_w = self.D_Sequencer.__getitem__(idx)
        return [M_batch_x,D_batch_x], D_batch_y, D_batch_w


class MinN_model(tf.keras.Model):

    def __init__(self, moleculeGNN, drugGNN, embedding_start = None,embedding_size = None):
        super(MinN_model, self).__init__()
        self.moleculeGNN = moleculeGNN
        self.drugGNN = drugGNN
        self.embedding_start = embedding_start
        self.embedding_end  = embedding_start + embedding_size

    def compile(self, *args, average_st_grads=False, **kwargs):
        """ Configures the model for learning.

        :param args: args inherited from Model.compile method. See source for details.
        :param average_st_grads: (bool) If True, net_state params are averaged wrt the number of iterations, summed otherwise.
        :param kwargs: Arguments supported for backwards compatibility only. Inherited from Model.compile method. See source for details.
        :raise: ValueError – In case of invalid arguments for 'optimizer', 'loss' or 'metrics'. """

        # force eager execution on super() model, since graph-mode must be implemented.
        run_eagerly = kwargs.pop("run_eagerly", False)

        super().compile(*args, **kwargs, run_eagerly=True)
        self.moleculeGNN.compile(*args, **kwargs, run_eagerly=run_eagerly)
        self.drugGNN.compile(*args, **kwargs, run_eagerly=run_eagerly)
        self._average_st_grads = average_st_grads

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


def weight(target, mu=1.0):
    w = 1/tf.math.reduce_sum(target, axis=0)
    normalize_w = w/ tf.math.reduce_min(w)
    return tf.math.pow(normalize_w,mu)