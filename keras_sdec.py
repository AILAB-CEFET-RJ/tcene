
'''
Codes originated from https://github.com/fferroni/DEC-Keras

I have implemented the custom loss function and the pairwise constaints mentioned in the paper
'Semi-supervised deep embedded clustering'. I also changed the optimizer used to trained the DEC.
'''
import sys
import numpy as np
import keras.backend as K
from keras.initializers import RandomNormal

from keras.layers import Layer
from tensorflow.keras.layers import InputSpec

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras import losses
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence

if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
import numpy as np


class SDEC:
    def __init__(self):
        pass

    @staticmethod
    def sdec_loss(add_loss):
        # Custom loss function combining KL divergence with an additional loss
        def loss_function(y_true, y_pred):
            kl_loss = KLDivergence()(y_true, y_pred)
            return kl_loss + add_loss
        return loss_function

    @staticmethod
    def add_loss(Z, Y, lambd=1e-5):
        # Additional loss function to impose constraints between data points
        n = Z.shape[0]
        a = SDEC.get_pairwise_constraints(Y)
        diff = Z[np.newaxis, :, :] - Z[:, np.newaxis, :]
        res = np.sum(a * np.sum(np.square(diff), axis=-1))
        return res * lambd / n

    @staticmethod
    def get_pairwise_constraints(Y):
        # Generate pairwise constraints matrix for loss calculation
        Y = Y.reshape(-1, 1)
        a = np.where(Y.dot(Y.T) == np.square(Y), 1, -1)
        return a


class ClusteringLayer(Layer):
    '''
    Clustering layer which converts latent space Z of input layer
    into a probability vector for each cluster defined by its centre in
    Z-space. Use Kullback-Leibler divergence as loss, with a probability
    target distribution.
    # Arguments
        output_dim: int > 0. Should be same as number of clusters.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        alpha: parameter in Student's t-distribution. Default is 1.0.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize weights and ensure input dimensions are correct
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]

        self.W = self.add_weight(
            shape=(self.initial_weights.shape),
            initializer=tf.constant_initializer(self.initial_weights),
            trainable=True,
            name='clustering_weights'
        )

    def call(self, x, mask=None):
        # Calculate the probabilities for cluster assignments using Student's t-distribution
        q = 1.0/(1.0 + tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(x, 1) - self.W), axis=2))**2 /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = tf.transpose(tf.transpose(q)/tf.reduce_sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        # Define the output shape of the layer
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # Configuration for rebuilding the layer
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self,
        n_clusters,
        input_dim,
        encoded=None,
        decoded=None,
        alpha=1.0, # probabilities for t-student distribution
        pretrained_weights=None,
        cluster_centres=None,
        batch_size=256,
        **kwargs):

        super(DeepEmbeddingClustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.encoded = encoded
        self.decoded = decoded
        self.alpha = alpha
        self.pretrained_weights = pretrained_weights
        self.cluster_centres = cluster_centres
        self.batch_size = batch_size

        self.learning_rate = 0.1
        self.iters_lr_update = 20000
        self.lr_change_rate = 0.1

        # greedy layer-wise training before end-to-end training:
        self.encoders_dims = [self.input_dim, 500, 500, 2000, 10] # Architecture 

        self.input_layer = Input(shape=(self.input_dim,), name='input')
        dropout_fraction = 0.2
        init_stddev = 0.01

        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i in range(1, len(self.encoders_dims)):
            
            encoder_activation = 'linear' if i == (len(self.encoders_dims) - 1) else 'relu'
            encoder = Dense(self.encoders_dims[i], activation=encoder_activation,
                            input_shape=(self.encoders_dims[i-1],),
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros', name='encoder_dense_%d'%i)
            self.encoders.append(encoder)

            decoder_index = len(self.encoders_dims) - i
            decoder_activation = 'linear' if i == 1 else 'relu'
            decoder = Dense(self.encoders_dims[i-1], activation=decoder_activation,
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros',
                            name='decoder_dense_%d'%decoder_index)
            self.decoders.append(decoder)

            autoencoder = Sequential([
                Dropout(dropout_fraction, input_shape=(self.encoders_dims[i-1],), 
                        name='encoder_dropout_%d'%i),
                encoder,
                Dropout(dropout_fraction, name='decoder_dropout_%d'%decoder_index),
                decoder
            ])
            autoencoder.compile(loss='mse', optimizer=SGD(learning_rate=self.learning_rate, momentum=0.9))
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for finetuning
        # Note that at this point dropout is discarded
        self.encoder = Sequential(self.encoders)
        self.encoder.compile(loss='mse', optimizer=SGD(learning_rate=self.learning_rate, momentum=0.9))
        self.decoders.reverse()
        self.autoencoder = Sequential(self.encoders + self.decoders)
        self.autoencoder.compile(loss='mse', optimizer=SGD(learning_rate=self.learning_rate, momentum=0.9))

        if cluster_centres is not None:
            assert cluster_centres.shape[0] == self.n_clusters
            assert cluster_centres.shape[1] == self.encoder.layers[-1].output_dim

        if self.pretrained_weights is not None:
            self.autoencoder.load_weights(self.pretrained_weights)

    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, X, save_autoencoder=False, layerwise_pretrain_iters=50000, finetune_iters=100000):
        if self.pretrained_weights is None: # Check if pretrained weights are available. If not, start training from scratch.

            # Calculate the number of iterations per epoch based on the batch size = 256 by default.
            iters_per_epoch = int(len(X) / self.batch_size) 
            
            # Determine the number of epochs for layer-wise and fine-tuning phases based on total iterations provided.
            layerwise_epochs = max(int(layerwise_pretrain_iters / iters_per_epoch), 1) 
            finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

            print('layerwise pretrain')
            # Start with the input data for the first layer of autoencoding.
            current_input = X
            
            # Update learning rate dynamically based on epochs.
            lr_epoch_update = max(1, self.iters_lr_update / float(iters_per_epoch))
            
            def step_decay(epoch): # Define a function for step decay learning rate schedule.
                initial_rate = self.learning_rate
                factor = int(epoch / lr_epoch_update)
                lr = initial_rate / (10 ** factor)
                return lr
            lr_schedule = LearningRateScheduler(step_decay)

            # Train each autoencoder layer-wise.
            for i, autoencoder in enumerate(self.layer_wise_autoencoders):
                if i > 0:
                    # For subsequent layers, use the output of the previous layer as input.
                    weights = self.encoders[i-1].get_weights()
                    dense_layer = Dense(self.encoders_dims[i], input_shape=(current_input.shape[1],),
                                        activation='relu',
                                        name='encoder_dense_copy_%d'%i) # FYI  TF 2.15 is not compatible with python 3.12+
                    encoder_model = Sequential([dense_layer])
                    
                    # Set weights to newly created dense layer and compile the model.
                    dense_layer.set_weights(weights)
                    encoder_model.compile(loss='mse', optimizer=SGD(learning_rate=self.learning_rate, momentum=0.9))
                    
                    # Predict the current input to feed into the next layer.
                    current_input = encoder_model.predict(current_input)

                # Fit the autoencoder on the current input and update the weights accordingly.
                autoencoder.fit(current_input, current_input, 
                                batch_size=self.batch_size, epochs=layerwise_epochs, callbacks=[lr_schedule])
                # Set trained weights to both encoder and decoder layers in the full model.
                self.autoencoder.layers[i].set_weights(autoencoder.layers[1].get_weights())
                self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(autoencoder.layers[-1].get_weights())
            
            print('Finetuning autoencoder')
            
            # Fine-tune the full autoencoder on the entire dataset.
            self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=finetune_epochs, callbacks=[lr_schedule])

            
            # Optionally save the autoencoder weights.
            if save_autoencoder:
                self.autoencoder.save_weights('autoencoder.h5')
        else:
            # If pretrained weights are available, load them instead of training from scratch.
            print('Loading pretrained weights for autoencoder.')
            self.autoencoder.load_weights(self.pretrained_weights)

        # After training, sync the weights across the encoder and autoencoder.
        # is this needed? Might be redundant...
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # If cluster centers are not pre-initialized, use K-Means to find initial centers.
        print('\nInitializing cluster centres with k-means.\n')
        if self.cluster_centres is None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            self.y_pred = kmeans.fit_predict(self.encoder.predict(X))
            self.cluster_centres = kmeans.cluster_centers_

        # Create the final DEC model, adding the clustering layer.
        self.DEC = Sequential([self.encoder,
                             ClusteringLayer(self.n_clusters,
                                                weights=self.cluster_centres,
                                                name='clustering')])
        # Compile the DEC model with a custom loss function and optimizer.
        sgd = SGD(learning_rate=0.01)
        self.DEC.compile(loss=SDEC.sdec_loss(add_loss=0), optimizer=sgd)
        return

    def cluster_acc(self, y_true, y_pred):
        import pdb; pdb.set_trace()
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max())+1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_assignment(w.max() - w)
        
        # Calculate the accuracy
        accuracy = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
        return accuracy, w
    
    def cluster(self, X, y=None,
                tol=0.01, update_interval=None,
                iter_max=1e6,
                save_interval=None,
                **kwargs):

        if update_interval is None:
            # 1 epochs
            update_interval = X.shape[0]/self.batch_size
        print('Update interval', update_interval)

        if save_interval is None:
            # 50 epochs
            save_interval = X.shape[0]/self.batch_size*50
        print('Save interval', save_interval)

        assert save_interval >= update_interval

        train = True
        iteration, index = 0, 0
        self.accuracy = []

        while train:
            sys.stdout.write('\r')
            # cutoff iteration
            if iter_max < iteration:
                print('Reached maximum iteration limit. Stopping training.')
                return self.y_pred

            # update (or initialize) probability distributions and propagate weight changes
            # from DEC model to encoder.
            if iteration % update_interval == 0:
                self.q = self.DEC.predict(X, verbose=0)
                self.p = self.p_mat(self.q)

                y_pred = self.q.argmax(1)
                delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0])
                if y is not None:
                    acc = self.cluster_acc(y, y_pred)[0]
                    self.accuracy.append(acc)
                    print('Iteration '+str(iteration)+', Accuracy '+str(np.round(acc, 5)))
                else:
                    print(str(np.round(delta_label*100, 5))+'%\ change in label assignment')

                if delta_label < tol:
                    print('Reached tolerance threshold. Stopping training.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred

                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.DEC.layers[0].layers[i].get_weights())
                self.cluster_centres = self.DEC.layers[-1].get_weights()[0]

            # train on batch
            sys.stdout.write('Iteration %d, ' % iteration)
            if (index+1)*self.batch_size > X.shape[0]:
                self.DEC.loss = SDEC.sdec_loss(
                    add_loss=SDEC.add_loss(self.encoder.predict(X[index*self.batch_size::]),
                                           y[index*self.batch_size::]))

                loss = self.DEC.train_on_batch(X[index*self.batch_size::], self.p[index*self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f' % loss)
            else:
                self.DEC.loss = SDEC.sdec_loss(
                    add_loss=SDEC.add_loss(self.encoder.predict(X[index*self.batch_size:(index+1) * self.batch_size]),
                                           y[index*self.batch_size:(index+1) * self.batch_size]))
                loss = self.DEC.train_on_batch(X[index*self.batch_size:(index+1) * self.batch_size],
                                               self.p[index*self.batch_size:(index+1) * self.batch_size])
                sys.stdout.write('Loss %f' % loss)
                index += 1


            # save intermediate
            if iteration % save_interval == 0:
                z = self.encoder.predict(X)
                pca = PCA(n_components=2).fit(z)
                z_2d = pca.transform(z)
                clust_2d = pca.transform(self.cluster_centres)
                # save states for visualization
                pickle.dump({'z_2d': z_2d, 'clust_2d': clust_2d, 'q': self.q, 'p': self.p},
                            open('c'+str(iteration)+'.pkl', 'wb'))
                # save DEC model checkpoints
                self.DEC.save('DEC_model_'+str(iteration)+'.h5')

            iteration += 1
            sys.stdout.flush()
        return
