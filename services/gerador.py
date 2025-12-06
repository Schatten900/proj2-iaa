# services/gerador.py
# -------------------------------------------------------------
# Gerador Pix2Pix U-Net para 128x128
# -------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers


def bloco_down(filtros, apply_batchnorm=True):
    bloco = tf.keras.Sequential()
    bloco.add(layers.Conv2D(filtros, 4, strides=2, padding="same", use_bias=False))

    if apply_batchnorm:
        bloco.add(layers.BatchNormalization())

    bloco.add(layers.LeakyReLU())
    return bloco


def bloco_up(filtros, apply_dropout=False):
    bloco = tf.keras.Sequential()
    bloco.add(layers.Conv2DTranspose(filtros, 4, strides=2, padding="same", use_bias=False))
    bloco.add(layers.BatchNormalization())
    
    if apply_dropout:
        bloco.add(layers.Dropout(0.5))

    bloco.add(layers.ReLU())
    return bloco


def construir_gerador():

    entradas = layers.Input(shape=[128, 128, 3])

    # Encoder 128 → 64 → 32 → ... → 1
    d1 = bloco_down(64, apply_batchnorm=False)(entradas)  
    d2 = bloco_down(128)(d1)
    d3 = bloco_down(256)(d2)
    d4 = bloco_down(512)(d3)
    d5 = bloco_down(512)(d4)
    d6 = bloco_down(512)(d5)

    # Decoder
    u1 = bloco_up(512, apply_dropout=True)(d6)
    u1 = layers.Concatenate()([u1, d5])

    u2 = bloco_up(512, apply_dropout=True)(u1)
    u2 = layers.Concatenate()([u2, d4])

    u3 = bloco_up(256)(u2)
    u3 = layers.Concatenate()([u3, d3])

    u4 = bloco_up(128)(u3)
    u4 = layers.Concatenate()([u4, d2])

    u5 = bloco_up(64)(u4)
    u5 = layers.Concatenate()([u5, d1])

    saidas = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(u5)

    return tf.keras.Model(inputs=entradas, outputs=saidas)
