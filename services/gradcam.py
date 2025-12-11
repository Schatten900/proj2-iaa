import tensorflow as tf
import numpy as np


def encontrar_ultima_conv(modelo):
    """
    Percorre o modelo inteiro, inclusive blocos Sequential,
    para encontrar a última camada Conv2D.
    """
    ultima_conv = None

    def explorar(layers):
        nonlocal ultima_conv
        for layer in layers:
            # se for Conv2D
            if isinstance(layer, tf.keras.layers.Conv2D):
                ultima_conv = layer

            # se for um bloco Sequential, explorar dentro
            if isinstance(layer, tf.keras.Model) or isinstance(layer, tf.keras.Sequential):
                explorar(layer.layers)

    explorar(modelo.layers)
    return ultima_conv


def gerar_gradcam(modelo, img_normalizada):

    entrada = tf.expand_dims(img_normalizada, axis=0)

    # achar a última camada conv2D
    ultima_conv = encontrar_ultima_conv(modelo)
    if ultima_conv is None:
        raise ValueError("Nenhuma camada Conv2D encontrada no modelo (nem dentro de Sequential).")

    # criar modelo que mapeia entrada -> (featuremap, saída final)
    grad_model = tf.keras.models.Model(
        inputs=modelo.inputs,
        outputs=[ultima_conv.output, modelo.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(entrada)
        loss = tf.reduce_mean(pred)

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(weights * conv_out[0], axis=-1)

    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)

    return cam
