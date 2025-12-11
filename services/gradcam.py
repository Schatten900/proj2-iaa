import tensorflow as tf
import numpy as np
import cv2

def gradcam(modelo, imagem, layer_name=None):

    if layer_name is None:
        for layer in reversed(modelo.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    grad_model = tf.keras.models.Model(
        inputs=modelo.inputs,
        outputs=[modelo.get_layer(layer_name).output, modelo.output]
    )

    img = np.expand_dims(imagem, axis=0)

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model([img, img])
        loss = tf.reduce_mean(pred)

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)

    cam = cam[0].numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (128, 128))

    return cam
    