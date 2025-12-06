# services/dataset.py
# -------------------------------------------------------------
# Dataset otimizado para CPU usando tf.data + map + cache
# Resolução reduzida para 128x128
# -------------------------------------------------------------

import os
import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 128


def carregar_imagem(caminho):
    """
    Carrega uma única imagem para inferência.
    Sem uso de tf.data, apenas cv2.
    """
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 127.5 - 1.0
    return img.astype("float32")

def carregar_imagem_cv2(caminho):
    caminho = caminho.decode("utf-8")
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 127.5 - 1.0
    return img.astype("float32")


def wrapper_processamento(caminho):
    img = tf.numpy_function(carregar_imagem_cv2, [caminho], tf.float32)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, img


def criar_dataset_treino(pasta="dataset/Healthy_Train50", batch_size=2):

    caminhos = tf.data.Dataset.list_files(pasta + "/*", shuffle=True)

    dataset = caminhos.map(
        wrapper_processamento,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.cache()
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
