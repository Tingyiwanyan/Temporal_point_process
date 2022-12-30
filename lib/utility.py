from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np

class projection_temporal(keras.layers.Layer):
    """
    define vector projection
    """
    def __init__(self, output_dim, **kwargs):
      self.output_dim = output_dim
      super(projection_temporal, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-2], self.output_dim),
                                      initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

    def call(self, input_data):

       return tf.keras.activations.relu(tf.math.multiply(input_data, self.kernel))


class translation_temporal(keras.layers.Layer):
    """
    define translation relation
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(translation_temporal, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-2], self.output_dim),
                                      initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
        #super(projection_temporal, self).build(input_shape)

    def call(self, input_data):
        input_data = tf.math.l2_normalize(input_data,axis=-1)
        return tf.math.add(input_data, self.kernel)

class att_temporal(keras.layers.Layer):
    """
    define self-attention layer
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(att_temporal, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_key = self.add_weight(name = 'kernel_key', shape = (input_shape[-1], self.output_dim),
                                      initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

        self.kernel_quary = self.add_weight(name = 'kernel_quary', shape = (input_shape[-1], self.output_dim),
                                      initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

    def call(self, input_data):
        conv_layers_outputs = []
        attention_outputs = []
        final_embedding_outputs = []
        self.check_att_output = attention_outputs
        self.check_conv_layer_output = conv_layers_outputs
        self.check_final_embedding = final_embedding_outputs
        #soft_max_layer = tf.keras.layers.Softmax()
        input_data = tf.math.l2_normalize(input_data,axis=-1)
        input_previous = input_data[:,0:-1,:,:]
        input_after = input_data[:,1:,:,:]
        self.check_input_previous = input_previous
        self.check_input_after = input_after
        att_output = tf.matmul(input_after,self.kernel_quary)
        att_output_key = tf.matmul(input_previous,self.kernel_key)
        #att_output = tf.math.exp(tf.matmul(att_output,tf.transpose(att_output_quary,perm=[0,1,3,2])))
        att_output = tf.matmul(att_output, tf.transpose(att_output_key, perm=[0, 1, 3, 2]))/10
        att_output = tf.keras.activations.softmax(att_output, axis=-1)
        #att_vis = att_output
        att_output = tf.expand_dims(att_output,axis=-1)
        self.check_att_output = att_output
        #input_data_value = tf.matmul(input_data,self.kernel_value)
        input_data_compare = tf.expand_dims(input_data,axis=2)[:,0:-1,:,:,:]
        self.check_input_data_compare = input_data_compare
        #self.check_input_data_value = input_data_value
        progression_embedding = tf.reduce_sum(tf.math.multiply(input_data_compare,att_output),axis=-2)
        self.check_progression_embedding = progression_embedding
        input_data_add = tf.gather(input_data,indices=list(np.array(range(input_data.shape[1]-1))+1),axis=1)
        input_data_add = tf.math.add(input_data_add,progression_embedding)
        self.check_input_data_add = input_data_add
        input_data_init = tf.gather(input_data,indices=[1],axis=1)
        self.check_input_data_init = input_data_init

        return [tf.concat([input_data_init,input_data_add],axis=1),att_output]

class feature_embedding_impotance(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(feature_embedding_impotance, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], self.output_dim),
                                      initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
        #super(projection_temporal, self).build(input_shape)

    def call(self, input_data):
        #soft_max_layer = tf.keras.layers.Softmax()
        final_embedding_att = tf.keras.activations.relu(tf.matmul(input_data, self.kernel))
        final_embedding_att = tf.keras.activations.softmax(tf.math.exp(final_embedding_att),axis=-2)
        final_embedding = tf.math.multiply(input_data, final_embedding_att)
        return [tf.reduce_sum(final_embedding, axis=-2),final_embedding_att]