import imp
from typing import List, Tuple, Union
import numpy as np
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Tuple, Union
import h5py
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.python.keras.saving import hdf5_format

from ms2deepscore import SpectrumBinner


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MHA(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MHA(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # # (batch_size, input_seq_len, d_model)
        # attn_output, _ = self.mha(x, x, x, mask)
        # attn_output = self.dropout1(attn_output, training=training)
        # # (batch_size, input_seq_len, d_model)
        # out1 = self.layernorm1(x + attn_output)

        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.dropout2(ffn_output, training=training)
        # # (batch_size, input_seq_len, d_model)
        # out2 = self.layernorm2(out1 + ffn_output)

        x_ = self.layernorm1(x)
        attn_output, _ = self.mha(x_, x_, x_, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = x + attn_output

        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(self.layernorm2(out1))
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = out1 + ffn_output

        return out2


class BaseConvLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BaseConvLayer, self).__init__()

        self.conv = tf.keras.layers.Conv1D(**kwargs)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        return tf.keras.activations.selu(self.bn(self.conv(x)))


class Net:
    def __init__(self,
                 spectrum_binner: SpectrumBinner, base_features_size, training=True, has_energy=False):

        # pylint: disable=too-many-arguments
        assert spectrum_binner.known_bins is not None, \
            "spectrum_binner does not contain known bins (run .fit_transform() on training data first!)"
        self.spectrum_binner = spectrum_binner
        self.base_features_size = base_features_size
        # Create base model
        self.base = self.get_base_model(
            base_features_size, training, has_energy)
        # Create head model
        self.model = self._get_head_model(input_dim=self.base_features_size,
                                          base_model=self.base)
        self.has_energy = has_energy

    def save(self, filename: Union[str, Path]):
        """
        Save model to file.

        Parameters
        ----------
        filename
            Filename to specify where to store the model.

        """
        with h5py.File(filename, mode='w') as f:
            hdf5_format.save_model_to_hdf5(self.model, f)
            f.attrs['spectrum_binner'] = self.spectrum_binner.to_json()

    @staticmethod
    def get_base_model(base_features_size, training, has_energy) -> keras.Model:

        token_len = 512
        d_model = 128
        dff = 512
        heads = 4
        strides = 32
        extra_bit = 2 if has_energy else 1
        model_input = Input(shape=(base_features_size), name='base_input')

        # token_len =100
        # d_model= int(base_features_size/token_len)
        # dff = 2048
        # heads = 8
        # model_layer = tf.keras.layers.Reshape((token_len, d_model))(model_input[:,extra_bit:])
        # model_layer = Dense(d_model, activation='relu')(model_layer)
        # model_layer = tf.keras.layers.LayerNormalization()(model_layer)
        model_layer = Dense(token_len*strides,
                            activation='relu')(model_input[:, extra_bit:])
        model_layer = tf.keras.layers.LayerNormalization()(model_layer)
        # model_layer_1 = Dense(8192, activation='relu')(model_input[:,0:1,:])
        # model_layer_1 = tf.keras.layers.LayerNormalization()(model_layer_1)
        # model_layer_2 = Dense(8192, activation='relu')(model_input[:,1:2,:])
        # model_layer_2 = tf.keras.layers.LayerNormalization()(model_layer_2)
        # model_layer_3 = Dense(8192, activation='relu')(model_input[:,2:3,:])
        # model_layer_3 = tf.keras.layers.LayerNormalization()(model_layer_3)
        # model_layer = tf.concat([model_layer_1, model_layer_2, model_layer_3], axis=1)

        model_layer = tf.expand_dims(model_layer, 2)
        # model_layer = Dense(energy_levels, activation='relu')(model_layer)
        model_layer_1 = tf.keras.layers.Conv1D(
            32, 32, strides=strides, padding="same",   input_shape=model_layer.shape[1:])(model_layer)
        model_layer_1 = tf.keras.layers.BatchNormalization()(model_layer_1)
        model_layer_1 = tf.keras.activations.relu(model_layer_1)

        model_layer_2 = tf.keras.layers.Conv1D(
            32, 32, strides=strides, padding="same",  input_shape=model_layer.shape[1:])(model_layer)
        model_layer_2 = tf.keras.layers.BatchNormalization()(model_layer_2)
        model_layer_2 = tf.keras.activations.relu(model_layer_2)

        model_layer_3 = tf.keras.layers.Conv1D(
            32, 64, strides=strides, padding="same",    input_shape=model_layer.shape[1:])(model_layer)
        model_layer_3 = tf.keras.layers.BatchNormalization()(model_layer_3)
        model_layer_3 = tf.keras.activations.relu(model_layer_3)

        model_layer_4 = tf.keras.layers.Conv1D(
            32, 128, strides=strides,  padding="same",    input_shape=model_layer.shape[1:])(model_layer)
        model_layer_4 = tf.keras.layers.BatchNormalization()(model_layer_4)
        model_layer_4 = tf.keras.activations.relu(model_layer_4)

        model_layer = tf.transpose(model_layer, [0, 2, 1])
        model_layer = Dense(token_len, activation='relu')(model_layer)
        model_layer_origin = tf.transpose(model_layer, [0, 2, 1])

        model_layer = tf.concat(
            [model_layer_1, model_layer_2, model_layer_3, model_layer_4,model_layer_origin], 2)
        # model_layer += model_layer_origin
        model_layer = Dense(d_model, activation='relu')(model_layer)
        # mass = tf.expand_dims(model_input[:, 0:1], 2)
        # mass = Dense(d_model, activation='relu')(mass)
        # mass = tf.keras.layers.LayerNormalization()(mass)
        extra_bit = 0
        # if has_energy:
        #     enegry = tf.expand_dims(model_input[:,1:2], 2)
        #     enegry = Dense(d_model, activation='relu')(mass)
        #     # enegry = tf.keras.layers.BatchNormalization()(enegry)
        #     model_layer = tf.keras.layers.Concatenate(axis=1)([mass,enegry,model_layer])
        # else:
        # model_layer = tf.keras.layers.Concatenate(axis=1)([mass,model_layer])

        pos_encoding = positional_encoding(token_len+extra_bit, d_model)
        model_layer += pos_encoding[:, :token_len+extra_bit, :]

        model_layer = EncoderLayer(d_model, heads, dff)(
            model_layer, training=training, mask=None)
        weight = tf.keras.layers.Conv1D(
            d_model, 8, activation='relu', padding="same",  input_shape=model_layer.shape[1:])(model_layer)
        model_layer = weight*model_layer

        model_layer = EncoderLayer(d_model, heads, dff)(
            model_layer, training=training, mask=None)
        # weight = tf.keras.layers.Conv1D(
        #     d_model, 8, activation='relu', padding="same",  input_shape=model_layer.shape[1:])(model_layer)
        # model_layer *= weight

        model_layer = EncoderLayer(d_model, heads, dff)(
            model_layer, training=training, mask=None)
        # weight = tf.keras.layers.Conv1D(
        #     d_model, 8, activation='relu', padding="same",  input_shape=model_layer.shape[1:])(model_layer)
        # model_layer *= weight

        model_layer = EncoderLayer(d_model, heads, dff)(
            model_layer, training=training, mask=None)
        model_layer = EncoderLayer(d_model, heads, dff)(
            model_layer, training=training, mask=None)
        model_layer = EncoderLayer(d_model, heads, dff)(
            model_layer, training=training, mask=None)
        model_layer = Dense(48, activation='relu')(model_layer)
        mass = tf.expand_dims(model_input[:, 0:1], 2)
        mass = tf.repeat(mass, repeats=token_len,axis=1)
        mass = tf.keras.layers.BatchNormalization()(mass)
        model_layer = tf.keras.layers.Concatenate(axis=2)([mass,model_layer])
        # model_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(model_layer)
        model_layer = tf.keras.layers.Flatten()(model_layer)
        # model_layer = Dense(1024, activation='relu')(model_layer)
        model_layer = Dense(500, activation='relu')(model_layer)
        return keras.Model(model_input, model_layer, name='base')

    @staticmethod
    def _get_head_model(input_dim: int,
                        base_model: keras.Model):
        input_a = Input(shape=input_dim, name="input_a")
        input_b = Input(shape=input_dim, name="input_b")
        embedding_a = base_model(input_a)
        embedding_b = base_model(input_b)
        cosine_similarity = keras.layers.Dot(axes=(1, 1),
                                             normalize=True,
                                             name="cosine_similarity")([embedding_a, embedding_b])
        return keras.Model(inputs=[input_a, input_b], outputs=[cosine_similarity],
                           name='head')

    def _construct_from_keras_model(self, keras_model):
        def valid_keras_model(given_model):
            assert given_model.layers, "Expected valid keras model as input."
            assert len(given_model.layers) > 2, "Expected more layers"
            assert len(
                keras_model.layers[2].layers) > 1, "Expected more layers for base model"

        valid_keras_model(keras_model)
        self.base = keras_model.layers[2]
        self.model = keras_model

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        self.model.fit_generator(*args, **kwargs)

    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

    def summary(self):
        self.base.summary()
        self.model.summary()

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)



