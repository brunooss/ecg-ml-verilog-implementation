#!/usr/bin/env python3
"""Versao Conv2D (H x 1) do modelo de Ribeiro et al.

Matematicamente IDENTICO ao reference/model.py, apenas reescrito com camadas 2D
para que o ONNX saia com `Conv` nativo 2D (NHWC) - o que o NNgen consome sem os
hacks de Unsqueeze/Squeeze que o tf2onnx insere para Conv1D (ver docs/05 secao 4).

Equivalencias:
    Conv1D(f, k)               -> Conv2D(f, (k, 1))
    MaxPooling1D(p)            -> MaxPooling2D((p, 1))
    entrada (4096, 12)         -> (4096, 1, 12)   [H, W, C]  (W=1)
Tudo o mais (BatchNorm, ReLU, Add residual, Flatten, Dense) e igual.
"""
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
import numpy as np


class ResidualUnit2D(object):
    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_keep_prob=0.8, kernel_size=16, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        if downsample > 1:
            y = MaxPooling2D((downsample, 1), strides=(downsample, 1), padding='same')(y)
        elif downsample < 1:
            raise ValueError("Number of samples should always decrease.")
        if n_filters_in != self.n_filters_out:
            y = Conv2D(self.n_filters_out, (1, 1), padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[3]           # canais no eixo C (ultimo)
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1a conv
        x = Conv2D(self.n_filters_out, (self.kernel_size, 1), padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        # 2a conv (com stride = downsample no eixo do tempo)
        x = Conv2D(self.n_filters_out, (self.kernel_size, 1), strides=(downsample, 1),
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])
            y = x
            x = self._batch_norm_plus_activation(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])
            x = Activation(self.activation_function)(x)
            y = x
        return [x, y]


def get_model_2d(n_classes, last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(4096, 1, 12), dtype=np.float32, name='signal')  # H,W,C
    x = signal
    x = Conv2D(64, (kernel_size, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnit2D(1024, 128, kernel_size=kernel_size,
                          kernel_initializer=kernel_initializer)([x, x])
    x, y = ResidualUnit2D(256, 196, kernel_size=kernel_size,
                          kernel_initializer=kernel_initializer)([x, y])
    x, y = ResidualUnit2D(64, 256, kernel_size=kernel_size,
                          kernel_initializer=kernel_initializer)([x, y])
    x, _ = ResidualUnit2D(16, 320, kernel_size=kernel_size,
                          kernel_initializer=kernel_initializer)([x, y])
    x = Flatten()(x)
    diagn = Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    return Model(signal, diagn)


if __name__ == "__main__":
    m = get_model_2d(6)
    m.summary()
