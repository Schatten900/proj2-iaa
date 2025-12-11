# services/discriminador.py
# -------------------------------------------------------------
# Discriminador PatchGAN modificado para 128x128
# -------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


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




def gerar_gradcam_discriminador(discriminador, img_in, img_out):

    entrada_in  = tf.expand_dims(img_in,  axis=0)
    entrada_out = tf.expand_dims(img_out, axis=0)

    # identificar última Conv2D no discriminador
    ultima_conv = None
    for layer in discriminador.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            ultima_conv = layer
            break

    if ultima_conv is None:
        raise ValueError("Discriminador não possui camada Conv2D.")

    grad_model = tf.keras.models.Model(
        inputs=discriminador.inputs,
        outputs=[ultima_conv.output, discriminador.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model([entrada_in, entrada_out])
        loss = tf.reduce_mean(pred)

    grads = tape.gradient(loss, conv_out)
    grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = tf.reduce_sum(grads * conv_out[0], axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)

    return cam
