import tensorflow as tf
import os
from services.gerador import construir_gerador

def carregar_modelo_pesos(caminho="gerador_treinado.h5"):
    if not os.path.exists(caminho):
        print("ERRO: arquivo de pesos não encontrado:", caminho)
        return None

    modelo = construir_gerador()
    modelo.build((None, 128, 128, 3))
    modelo.load_weights(caminho)

    print("✔ Pesos carregados com sucesso!")
    return modelo
