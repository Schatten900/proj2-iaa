# services/treinamento.py
# -------------------------------------------------------------
# Treinamento otimizado para CPU, validação só no final
# -------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

from services.dataset import criar_dataset_treino
from services.gerador import construir_gerador
from services.discriminador import construir_discriminador


LR = 2e-4
LAMBDA_L1 = 50     # menor = treinamento mais rápido
EPOCHS = 60
BATCH_SIZE = 2


def perda_discriminador(real_output, fake_output):
    BCE = losses.BinaryCrossentropy(from_logits=True)
    return 0.5 * (BCE(tf.ones_like(real_output), real_output) +
                  BCE(tf.zeros_like(fake_output), fake_output))


def perda_gerador_gan(fake_output):
    BCE = losses.BinaryCrossentropy(from_logits=True)
    return BCE(tf.ones_like(fake_output), fake_output)


def perda_l1(real, fake):
    return tf.reduce_mean(tf.abs(real - fake))


opt_g = Adam(LR, beta_1=0.5)
opt_d = Adam(LR, beta_1=0.5)


@tf.function
def treinar_batch(gerador, discriminador, real_in, real_tgt):

    with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:

        fake_img = gerador(real_in, training=True)

        disc_real = discriminador([real_in, real_tgt], training=True)
        disc_fake = discriminador([real_in, fake_img], training=True)

        g_gan = perda_gerador_gan(disc_fake)
        g_l1 = perda_l1(real_tgt, fake_img)

        g_total = g_gan + LAMBDA_L1 * g_l1

        d_total = perda_discriminador(disc_real, disc_fake)

    grad_g = tape_g.gradient(g_total, gerador.trainable_variables)
    grad_d = tape_d.gradient(d_total, discriminador.trainable_variables)

    opt_g.apply_gradients(zip(grad_g, gerador.trainable_variables))
    opt_d.apply_gradients(zip(grad_d, discriminador.trainable_variables))

    return g_total, d_total


def treinar():

    dataset = criar_dataset_treino(batch_size=BATCH_SIZE)

    gerador = construir_gerador()
    discriminador = construir_discriminador()

    print("\n=========== INICIANDO TREINAMENTO ===========")

    for ep in range(1, EPOCHS + 1):

        g_loss_ep = 0
        d_loss_ep = 0
        batches = 0

        for real_in, real_tgt in dataset:
            g_loss, d_loss = treinar_batch(gerador, discriminador, real_in, real_tgt)
            g_loss_ep += g_loss
            d_loss_ep += d_loss
            batches += 1

        print(f"Época {ep}/{EPOCHS} | "
              f"G_Loss = {g_loss_ep/batches:.4f} | "
              f"D_Loss = {d_loss_ep/batches:.4f}")

    print("\n=========== TREINAMENTO FINALIZADO ===========\n")

    gerador.save("gerador_treinado.h5")
    print("Modelo salvo como gerador_treinado.h5")

    return gerador


if __name__ == "__main__":
    treinar()
