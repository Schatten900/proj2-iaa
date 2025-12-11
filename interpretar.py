import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from carregar_modelo import carregar_modelo_pesos
from services.dataset import carregar_imagem
from services.metrics import mapa_deltaE_ciede2000
from services.discriminador import gerar_gradcam_discriminador
from services.discriminador import construir_discriminador
from inferencia import desnormalizar


os.makedirs("interpretabilidade", exist_ok=True)


def interpretar_imagem(caminho_img, gerador, idx):

    # imagem real normalizada [-1,1]
    real_norm = carregar_imagem(caminho_img)
    real = desnormalizar(real_norm)

    entrada = np.expand_dims(real_norm, axis=0)

    # reconstrução
    fake_norm = gerador(entrada, training=False)[0].numpy()
    fake = desnormalizar(fake_norm)

    # ΔE2000
    deltaE = mapa_deltaE_ciede2000(real, fake)

    # Grad-CAM do DISCRIMINATOR
    discriminador = construir_discriminador()
    cam = gerar_gradcam_discriminador(discriminador, real_norm, fake_norm)
    cam = cv2.resize(cam, (128, 128))

    # heatmap
    heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.5, real, 0.5, 0)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    ax[0].imshow(real)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(fake)
    ax[1].set_title("Reconstruída")
    ax[1].axis("off")

    ax[2].imshow(deltaE, cmap="inferno")
    ax[2].set_title("ΔE2000")
    ax[2].axis("off")

    ax[3].imshow(cam, cmap="jet")
    ax[3].set_title("Grad-CAM (Discriminador)")
    ax[3].axis("off")

    ax[4].imshow(overlay)
    ax[4].set_title("Overlay")
    ax[4].axis("off")

    nome = f"interpretabilidade/figura_{idx}.png"
    plt.savefig(nome, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✔ Figura salva: {nome}")


def executar_interpretabilidade():

    gerador = carregar_modelo_pesos()

    imagens = (
        glob.glob("dataset/Healthy_Test50/*")[:2] +
        glob.glob("dataset/Disease_Test100/*")[:2]
    )

    print(f"\nRodando interpretabilidade para {len(imagens)} imagens...\n")

    for i, img in enumerate(imagens):
        interpretar_imagem(img, gerador, i)


if __name__ == "__main__":
    executar_interpretabilidade()
