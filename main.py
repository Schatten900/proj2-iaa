# main.py
#
# Menu principal para:
#   1. Treinar o modelo
#   2. Realizar inferência em imagem aleatória do dataset de teste
#

import os
import random

from treinamento import treinar
from inferencia import inferir

# Pastas do dataset de teste
PASTAS_TESTE = [
    "dataset/Disease_Test100",
    "dataset/Healthy_Test50"
]


def escolher_imagem_aleatoria():
    """
    Seleciona uma imagem aleatória do dataset de teste.
    """
    pasta = random.choice(PASTAS_TESTE)

    imagens = [
        arq for arq in os.listdir(pasta)
        if arq.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not imagens:
        raise RuntimeError(f"Nenhuma imagem encontrada na pasta {pasta}")

    nome = random.choice(imagens)
    caminho = os.path.join(pasta, nome)

    print(f"\n🌿 Imagem sorteada para inferência: {caminho}\n")
    return caminho


def executar_inferencia():
    imagem = escolher_imagem_aleatoria()
    resultado = inferir(imagem)   # <-- removido checkpoint_dir
    print(f"\nResultado Final da Inferência: {resultado}\n")



def executar_treinamento():
    """
    Executa o treinamento do modelo.
    """
    print("\nIniciando treinamento...\n")
    treinar()
    print("\nTreinamento finalizado!\n")


def main():
    while True:
        print("\n==============================")
        print("     SISTEMA PIX2PIX IA")
        print("==============================")
        print("1 - Treinar modelo")
        print("2 - Inferir imagem aleatória do teste")
        print("3 - Sair")
        print("==============================\n")

        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            executar_treinamento()

        elif opcao == "2":
            executar_inferencia()

        elif opcao == "3":
            print("Encerrando...")
            break

        else:
            print("Opção inválida!")
    exit()

if __name__ == "__main__":
    main()
