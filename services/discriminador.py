# services/discriminador.py
# -------------------------------------------------------------
# Discriminador PatchGAN modificado para 128x128
# -------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers


def bloco_disc(filtros, stride=2):
    bloco = tf.keras.Sequential()
    bloco.add(layers.Conv2D(filtros, 4, strides=stride, padding="same"))
    bloco.add(layers.BatchNormalization())
    bloco.add(layers.LeakyReLU())
    return bloco


def construir_discriminador():

    inp = layers.Input(shape=[128, 128, 3], name="input_image")
    tar = layers.Input(shape=[128, 128, 3], name="target_image")

    x = layers.concatenate([inp, tar])

    d1 = bloco_disc(64, stride=2)(x)
    d2 = bloco_disc(128, stride=2)(d1)
    d3 = bloco_disc(256, stride=2)(d2)
    d4 = bloco_disc(512, stride=1)(d3)

    saida = layers.Conv2D(1, 4, strides=1, padding="same")(d4)

    return tf.keras.Model(inputs=[inp, tar], outputs=saida)
