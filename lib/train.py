from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
from lib.utility import projection_temporal as projection
from lib.utility import translation_temporal as translation
from lib.utility import att_temporal
from lib.utility import feature_embedding_impotance
semantic_step_global = 6
semantic_positive_sample = 4
unsupervised_cluster_num = 10
latent_dim_global = 100
positive_sample_size = 10
batch_size = 128
unsupervised_neg_size = 5
reconstruct_resolution = 7
feature_num = 34


class temporal_point_process():
    def __init__(self):
        self.projection_model = projection(latent_dim_global)
        self.relation_layer = translation(latent_dim_global)
        self.att_relation_layer = att_temporal(latent_dim_global)
        self.embedding_att_layer = feature_embedding_impotance(1)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.ave_all = [8.38230435e+01, 9.75000000e+01, 3.69060000e+01, 1.18333333e+02,
                        7.71140148e+01, 5.90000000e+01, 1.81162791e+01, 0.00000000e+00,
                        -2.50000000e-01, 2.43333333e+01, 5.04195804e-01, 7.38666667e+00,
                        4.00504808e+01, 9.60000000e+01, 4.20000000e+01, 1.65000000e+01,
                        7.70000000e+01, 8.35000000e+00, 1.06000000e+02, 9.00000000e-01,
                        1.16250000e+00, 1.25333333e+02, 1.65000000e+00, 2.00000000e+00,
                        3.36666667e+00, 4.08000000e+00, 7.00000000e-01, 3.85000000e+00,
                        3.09000000e+01, 1.05000000e+01, 3.11000000e+01, 1.08333333e+01,
                        2.55875000e+02, 1.93708333e+02]

        self.std_all = [1.40828962e+01, 2.16625304e+00, 5.53108392e-01, 1.66121889e+01,
                        1.08476132e+01, 9.94962122e+00, 3.59186362e+00, 0.00000000e+00,
                        3.89407506e+00, 3.91858658e+00, 2.04595954e-01, 5.93467422e-02,
                        7.72257867e+00, 8.87388075e+00, 5.77276895e+02, 1.79879091e+01,
                        1.36508822e+02, 6.95188900e-01, 5.09788015e+00, 1.43347221e+00,
                        3.75415153e+00, 4.03968485e+01, 1.71418146e+00, 3.15505742e-01,
                        1.17084555e+00, 4.77914796e-01, 3.62933460e+00, 9.91058703e+00,
                        4.60374699e+00, 1.64019340e+00, 1.68795640e+01, 6.23941196e+00,
                        1.75014175e+02, 1.03316340e+02]

        """
        define hyper-parameters
        """
        self.gaussian_mu = 0
        self.gaussian_sigma = 0.0001
        self.batch_size = batch_size
        self.vital_length = 8
        self.lab_length = 19
        self.blood_length = 27
        self.epoch = 20
        self.feature_num = 34
        self.pre_train_epoch = 7
        self.latent_dim = latent_dim_global
        self.tau = 1
        self.time_sequence = 48#self.read_d.time_sequence
        self.start_sampling_index = 5
        self.sampling_interval = 5

        self.create_memory_bank()
        self.length_train = len(self.train_data)
        self.steps = self.length_train // self.batch_size

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0003,
            decay_steps=self.steps,
            decay_rate=0.3)

    def create_memory_bank(self):
        """
        Loading data
        """

        file_path = '/home/tingyi/Downloads/Interpolate_data/'
        with open(file_path + 'train.npy', 'rb') as f:
            self.train_data = np.load(f)
        with open(file_path + 'train_logit.npy', 'rb') as f:
            self.train_logit = np.load(f)
        with open(file_path + 'train_on_site_time.npy', 'rb') as f:
            self.train_on_site_time = np.load(f)
        with open(file_path + 'val.npy', 'rb') as f:
            self.val_data = np.load(f)
        with open(file_path + 'val_logit.npy', 'rb') as f:
            self.val_logit = np.load(f)
        with open(file_path + 'val_on_site_time.npy', 'rb') as f:
            self.val_on_site_time = np.load(f)

        with open(file_path + 'train_origin.npy', 'rb') as f:
            self.train_data_origin = np.load(f)
        """
        self.train_data = np.expand_dims(self.train_data, axis=3)
        self.val_data = np.expand_dims(self.val_data, axis=3)
        self.max_train_data = np.max(np.reshape(self.train_data, (self.train_data.shape[0] * self.train_data.shape[1],
                                                                  self.train_data.shape[2])), 0)
        self.min_train_data = np.min(np.reshape(self.train_data, (self.train_data.shape[0] * self.train_data.shape[1],
                                                                  self.train_data.shape[2])), 0)

        self.index_train = np.array(range(self.train_data.shape[0]))

        self.train_data_range = self.max_train_data - self.min_train_data
        for i in range(self.train_data_range.shape[0]):
            if self.train_data_range[i] == 0:
                self.train_data_range[i] = 1

        self.train_data_norm = (np.reshape(self.train_data,
                                           (self.train_data.shape[0] * self.train_data.shape[1],
                                            self.train_data.shape[2])) - self.min_train_data) \
                               / self.train_data_range

        self.train_data_norm = np.reshape(self.train_data_norm, (self.train_data.shape[0], self.train_data.shape[1],
                                                                 self.train_data.shape[2]))
        """
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_data, self.train_logit, self.train_on_site_time, self.train_data))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        cohort_index = np.where(self.train_logit == 1)[0]
        control_index = np.where(self.train_logit == 0)[0]
        self.memory_bank_cohort = self.train_data[cohort_index, :, :]
        self.memory_bank_control = self.train_data[control_index, :, :]
        self.memory_bank_cohort_on_site = self.train_on_site_time[cohort_index]
        self.memory_bank_control_on_site = self.train_on_site_time[control_index]
        self.memory_bank_cohort_origin = self.train_data_origin[cohort_index]
        self.memory_bank_control_origin = self.train_data_origin[control_index]
        self.num_cohort = self.memory_bank_cohort.shape[0]
        self.num_control = self.memory_bank_control.shape[0]


    def temporal_progression_model(self):
        inputs = layers.Input((self.time_sequence, self.feature_num))
        #inputs = tf.expand_dims(inputs, axis=3)
        output = self.projection_model(inputs)
        forward_1 = layers.Dense(
            self.latent_dim,
            # use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )
        forward_2 = layers.Dense(
            self.latent_dim,
            # use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            activation='relu'
        )
        #normalization = tfa.layers.WeightNormalization()
        output = forward_1(output)
        output = forward_2(output)
        self.check_output_single = output
        output = self.relation_layer(output)
        [output_whole,att_temporal] = self.att_relation_layer(output)
        #output_whole = tfa.layers.WeightNormalization(output_whole)
        self.check_output_whole = output_whole
        [output,att_final] = self.embedding_att_layer(output_whole)

        return tf.keras.Model(inputs,[output,att_temporal,att_final])

    def train_temporal_progression(self):
        # input = layers.Input((self.time_sequence, self.feature_num))
        #self.tcn = self.tcn_encoder_second_last_level()

        self.temporal = self.temporal_progression_model()
        # tcn = self.tcn(input)
        self.auc_all = []
        self.loss_track = []
        # self.model_extractor = tf.keras.Model(input, tcn, name="time_extractor")
        self.projection_layer = self.project_logit()
        self.bceloss = tf.keras.losses.BinaryCrossentropy()

        for epoch in range(self.pre_train_epoch):
            print("\nStart of epoch %d" % (epoch,))

            # extract_val, global_val,k = self.model_extractor(self.val_data)
            last_layer_output_val = self.temporal(self.val_data)[0]
            #last_layer_output_val = tcn_temporal_output_val[1]
            on_site_extract_val = [last_layer_output_val[i, np.abs(int(self.val_on_site_time[i]) - 1), :] for i in
                                   range(self.val_on_site_time.shape[0])]
            on_site_extract_array_val = tf.stack(on_site_extract_val)
            prediction_val = self.projection_layer(on_site_extract_array_val)
            self.check_prediction_val = prediction_val
            val_acc = roc_auc_score(self.val_logit, prediction_val)
            print("auc")
            print(val_acc)
            self.auc_all.append(val_acc)
            for step, (x_batch_train, y_batch_train, on_site_time, x_batch_origin) in enumerate(self.train_dataset):
                self.check_x_batch = x_batch_train
                self.check_on_site_time = on_site_time
                self.check_label = y_batch_train
                with tf.GradientTape() as tape:
                    last_layer_output = self.temporal(x_batch_train)[0]
                    self.check_output = last_layer_output
                    #last_layer_output = tcn_temporal_output[1]
                    on_site_extract = [last_layer_output[i, int(on_site_time[i] - 1), :] for i in
                                       range(on_site_time.shape[0])]
                    on_site_extract_array = tf.stack(on_site_extract)
                    prediction = self.projection_layer(on_site_extract_array)
                    loss = self.bceloss(y_batch_train, prediction)
                    self.check_prediction = prediction

                gradients = \
                    tape.gradient(loss,
                                  self.temporal.trainable_variables + self.projection_layer.trainable_weights)
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

                optimizer.apply_gradients(zip(gradients,
                                              self.temporal.trainable_variables + self.projection_layer.trainable_weights))

                if step % 20 == 0:
                    print("Training loss(for one batch) at step %d: %.4f"
                          % (step, float(loss)))
                    print("seen so far: %s samples" % ((step + 1) * self.batch_size))

                    self.loss_track.append(loss)

    def transition_project_layer(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                # layers.Input((50)),
                layers.Dense(
                    self.latent_dim,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='sigmoid'
                )
            ],
            name="transition_projection",
        )
        return model

    def project_logit(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                layers.Input((self.latent_dim)),
                layers.Dense(
                    50,
                    # use_bias=False,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='relu'
                ),
                layers.Dense(
                    1,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                    activation='sigmoid'
                )
            ],
            name="projection_logit",
        )
        return model



