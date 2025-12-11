import glob
import numpy as np

from inferencia import carregar_modelo, desnormalizar
from services.dataset import carregar_imagem
from services.metrics import mapa_deltaE_ciede2000

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LIMIAR = 3.325781

def avaliar():

    saudaveis = glob.glob("dataset/Healthy_Test50/*")
    doentes   = glob.glob("dataset/Disease_Test100/*")

    print(f"Saudáveis: {len(saudaveis)}")
    print(f"Doentes: {len(doentes)}")

    if len(saudaveis) == 0 and len(doentes) == 0:
        print("Nenhuma imagem encontrada.")
        return

    gerador = carregar_modelo()

    y_true = []
    y_pred = []

    todos = saudaveis + doentes
    labels = [0]*len(saudaveis) + [1]*len(doentes)

    for img_path, true_label in zip(todos, labels):

        real_norm = carregar_imagem(img_path)
        real = desnormalizar(real_norm)

        fake_norm = gerador(np.expand_dims(real_norm, axis=0))[0].numpy()
        fake = desnormalizar(fake_norm)

        deltaE = mapa_deltaE_ciede2000(real, fake)
        erro = np.mean(deltaE)

        pred = 1 if erro > LIMIAR else 0

        y_true.append(true_label)
        y_pred.append(pred)

    if len(y_true) == 0:
        print("Nenhuma imagem válida processada.")
        return

    print("\n MÉTRICAS:\n")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
    print("F1-score :", f1_score(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    avaliar()
