from tensorflow import keras
import tensorflow as tf
from blocks import Encoder, Decoder
from ml_util import create_padding_mask, create_look_ahead_mask


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, rate)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        return dec_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = None

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = None

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        look_ahead_mask = tf.expand_dims(look_ahead_mask, 0)
        look_ahead_mask = tf.repeat(look_ahead_mask, tf.shape(tar)[0], axis=0)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask
