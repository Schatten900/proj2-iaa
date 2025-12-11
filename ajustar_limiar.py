import glob
import numpy as np
from inferencia import carregar_modelo, desnormalizar
from services.dataset import carregar_imagem
from services.metrics import mapa_deltaE_ciede2000

def ajustar():

    saudaveis = glob.glob("dataset/Healthy_Test50/*")
    doentes = glob.glob("dataset/Disease_Test100/*")

    gerador = carregar_modelo()

    erros_saudaveis = []
    erros_doentes = []

    for img_path in saudaveis:
        real_norm = carregar_imagem(img_path)
        real = desnormalizar(real_norm)
        fake = gerador(np.expand_dims(real_norm,0))[0].numpy()
        fake = desnormalizar(fake)
        deltaE = mapa_deltaE_ciede2000(real, fake)
        erros_saudaveis.append(np.mean(deltaE))

    for img_path in doentes:
        real_norm = carregar_imagem(img_path)
        real = desnormalizar(real_norm)
        fake = gerador(np.expand_dims(real_norm,0))[0].numpy()
        fake = desnormalizar(fake)
        deltaE = mapa_deltaE_ciede2000(real, fake)
        erros_doentes.append(np.mean(deltaE))

    print("\nMédias:")
    print("Saudáveis:", np.mean(erros_saudaveis))
    print("Doentes  :", np.mean(erros_doentes))

    print("\nSugestão de limiar:")
    print((np.mean(erros_saudaveis)+np.mean(erros_doentes))/2)

if __name__ == "__main__":
    ajustar()
