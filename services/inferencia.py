# inferir.py
#
# Carrega o gerador salvo em "gerador_treinado.h5"
# e realiza inferência e classificação.

import os
import cv2
import numpy as np
import tensorflow as tf

from services.dataset import carregar_imagem
from services.gerador import construir_gerador


# -------------------------------------------------------------
# 1. Carregar modelo .h5 treinado
# -------------------------------------------------------------
def carregar_modelo(caminho_modelo="gerador_treinado.h5"):
    if not os.path.exists(caminho_modelo):
        print("[ERRO] Modelo gerador_treinado.h5 não encontrado! Treine o modelo primeiro.")
        return None

    print(f"[OK] Carregando modelo: {caminho_modelo}")
    modelo = tf.keras.models.load_model(caminho_modelo, compile=False)
    return modelo


# -------------------------------------------------------------
# 2. Realiza inferência
# -------------------------------------------------------------
def gerar_reconstrucao(gerador, img_path):
    img = carregar_imagem(img_path)          # [-1,1]
    img = np.expand_dims(img, axis=0)        # (1,128,128,3)
    gerada = gerador(img, training=False)
    return gerada[0].numpy()


def desnormalizar(img):
    img = (img + 1.0) * 127.5
    return np.clip(img, 0, 255).astype("uint8")


def calcular_diferenca(real, gerada):
    return np.abs(real - gerada)


def classificar(real, gerada, limiar=25):
    diff = calcular_diferenca(real, gerada)
    erro = np.mean(diff)
    print(f"Erro médio = {erro:.2f}")

    return "Folha DOENTE" if erro > limiar else "Folha Saudável"


# -------------------------------------------------------------
# 3. Função principal
# -------------------------------------------------------------
def inferir(img_path, salvar=True):
    gerador = carregar_modelo()
    if gerador is None:
        return None

    real_norm = carregar_imagem(img_path)
    real = desnormalizar(real_norm)

    gerada_norm = gerar_reconstrucao(gerador, img_path)
    gerada = desnormalizar(gerada_norm)

    diff = calcular_diferenca(real, gerada)
    classificacao = classificar(real, gerada)

    if salvar:
        os.makedirs("resultados_inferencia", exist_ok=True)
        base = os.path.basename(img_path)

        cv2.imwrite(f"resultados_inferencia/real_{base}", cv2.cvtColor(real, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"resultados_inferencia/gerada_{base}", cv2.cvtColor(gerada, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"resultados_inferencia/diff_{base}", diff)

        print("Resultados salvos em resultados_inferencia/")

    return classificacao
