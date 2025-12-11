# inferir.py
#
# Carrega o gerador salvo em "gerador_treinado.h5"
# e realiza inferência e classificação.

import os
import cv2
import numpy as np
import tensorflow as tf

from services.dataset import carregar_imagem
from services.metrics import mapa_deltaE_ciede2000


def carregar_modelo(path="gerador_treinado.h5"):
    return tf.keras.models.load_model(path, compile=False)

def desnormalizar(img):
    img = (img + 1) * 127.5
    return np.clip(img, 0, 255).astype(np.uint8)
    
def inferir(img_path, limiar=25):

    gerador = carregar_modelo()

    real_norm = carregar_imagem(img_path)
    real = desnormalizar(real_norm)

    fake_norm = gerador(np.expand_dims(real_norm, axis=0))[0].numpy()
    fake = desnormalizar(fake_norm)

    deltaE = mapa_deltaE_ciede2000(real, fake)
    erro = np.mean(deltaE)

    classe = "Folha DOENTE" if erro > limiar else "Folha SAUDÁVEL"

    os.makedirs("resultados_inferencia", exist_ok=True)
    nome = os.path.basename(img_path)

    cv2.imwrite(f"resultados_inferencia/real_{nome}", cv2.cvtColor(real, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"resultados_inferencia/gerada_{nome}", cv2.cvtColor(fake, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"resultados_inferencia/deltaE_{nome}", deltaE)

    print("Classificação:", classe)
    print("Erro médio:", erro)

    return classe