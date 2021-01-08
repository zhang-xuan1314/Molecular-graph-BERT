import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt



def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


# def get_angles(pos, i, d_model):
#   angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
#   return pos * angle_rates
#
#
# def positional_encoding(position, d_model):
#     angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                             np.arange(d_model)[np.newaxis, :],
#                             d_model)
#
#     # apply sin to even indices in the array; 2i
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#
#     # apply cos to odd indices in the array; 2i+1
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#
#     pos_encoding = angle_rads[np.newaxis, ...]
#
#     return tf.cast(pos_encoding, dtype=tf.float32)


# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#
#     # add extra dimensions to add the padding
#     # to the attention logits.
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
#
# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask,adjoin_matrix):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask,adjoin_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,adjoin_matrix)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=gelu),  # (batch_size, seq_len, dff)tf.keras.layers.LeakyReLU(0.01)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        attn_output, attention_weights = self.mha(x, x, x, mask,adjoin_matrix)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2,attention_weights


# class EmbeddingDense(tf.keras.layers.Layer):
#     """运算跟Dense一致，只不过kernel用Embedding层的embedding矩阵
#     """
#
#     def __init__(self,embedding_layer, activation=None, **kwargs):
#         super(EmbeddingDense, self).__init__(**kwargs)
#         self.activation = activation
#         self.units = embedding_layer.input_dim
#         self.embedding_layer = embedding_layer
#         self.activation = tf.keras.layers.Activation(self.activation)
#
#
#     def build(self, input_shape):
#         super(EmbeddingDense, self).build(input_shape)
#         self.kernel = tf.transpose(self.embedding_layer.embeddings)
#         self.bias = self.add_weight(name='bias',
#                                     shape=(self.units,),
#                                     initializer='zeros')
#
#     def call(self, inputs):
#         outputs = tf.matmul(inputs, self.kernel)
#         outputs = outputs+self.bias
#         outputs = self.activation(outputs)
#         return outputs
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[:-1] + (self.units,)


class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding,
        #                                         self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
        return x  # (batch_size, input_seq_len, d_model)

class Encoder_test(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder_test, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding,
        #                                         self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        attention_weights_list = []
        xs = []

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
            attention_weights_list.append(attention_weights)
            xs.append(x)

        return x,attention_weights_list,xs

class BertModel_test(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 17,dropout_rate = 0.1):
        super(BertModel_test, self).__init__()
        self.encoder = Encoder_test(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
    def call(self,x,adjoin_matrix,mask,training=False):
        x,att,xs = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x,att,xs




class BertModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 17,dropout_rate = 0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x


class PredictModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size =17,dropout_rate = 0.1,dense_dropout=0.1):
        super(PredictModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(256,activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = x[:,0,:]
        x = self.fc1(x)
        x = self.dropout(x,training=training)
        x = self.fc2(x)
        return x



class PredictModel_test(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size =17,dropout_rate = 0.1,dense_dropout=0.5):
        super(PredictModel_test, self).__init__()
        self.encoder = Encoder_test(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self,x,adjoin_matrix,mask,training=False):
        x,att,xs = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x,att,xs






