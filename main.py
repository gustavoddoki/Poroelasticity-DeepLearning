import tensorflow as tf
import numpy as np
import pandas as pd

from CreateModel import create_model
from ComputeLossFunction import compute_loss
from CreateSample import create_sample


def train_step(L, t_f, E, K, model, optimizer, batch_vars, random_vars, batch_size, real_solution):
    with tf.GradientTape() as tape:
        loss_total, loss_componentes, loss_real = compute_loss(L, t_f, E, K, model, batch_vars, random_vars, batch_size, real_solution)
    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_total, loss_componentes, loss_real


def main():

    L = 0.5
    t_f = 1
  
    E = 1
    K = 1
  
    size_x = 51
    size_t = 101
    batch_size = 256
    neurons_per_layer = 128
    model = create_model(size_x, size_t, neurons_per_layer)
  
    epoch = 0
  
    learning_rate = 1e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate)
  
    dict_treinamento = {'epoch': [], 'loss total': [], 'loss equação deslocamento': [], 'loss equação pressão': [],
                        'loss contorno direito': [], 'loss contorno esquerdo': [], 'loss condição inicial': [], 'du': [], 'dp': []}
  
    x_domain = np.linspace(0, L, num=size_x, dtype=np.float64)
    t_domain = np.linspace(0, t_f, num=size_t, dtype=np.float64).reshape((size_t, 1))
    x_domain = np.tile(x_domain, (size_t, 1))
  
    melhor_loss = float('inf')
  
    step_size = 10
    max_loglr = -3
    min_loglr = -7
  
    delta_loglr = max_loglr - min_loglr
    var_loglr = delta_loglr / step_size
  
    learning_rate = 10 ** max_loglr
    cycle_counter = 1
    factor = 1
  
    limite_epoch = 100000
    
    while True:
  
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        epoch += 1
  
        batch_vars, random_vars, real_solution = create_sample(L, t_f, E, K, size_x, size_t, batch_size, x_domain, t_domain)
        loss_total, loss_componentes, loss_real = train_step(L, t_f, E, K, model, optimizer, batch_vars, random_vars, batch_size, real_solution)
  
        dict_treinamento['epoch'].append(epoch)
        dict_treinamento['loss total'].append(float(loss_total))
        dict_treinamento['loss equação deslocamento'].append(float(loss_componentes[0]))
        dict_treinamento['loss equação pressão'].append(float(loss_componentes[1]))
        dict_treinamento['loss contorno direito'].append(float(loss_componentes[2]))
        dict_treinamento['loss contorno esquerdo'].append(float(loss_componentes[3]))
        dict_treinamento['loss condição inicial'].append(float(loss_componentes[4]))
        dict_treinamento['du'].append(float(loss_real[0]))
        dict_treinamento['dp'].append(float(loss_real[1]))
  
        if loss_total < melhor_loss:
            melhor_loss = loss_total
            print(epoch, melhor_loss, loss_real[0], loss_real[1])
            model.save('model_best')
  
        if epoch % 1000 == 0:
            print({chave: lista[-1] for chave, lista in dict_treinamento.items()})
  
            with open('dados_treinamento.csv', mode='a+', encoding='utf-8') as f:
                for i in range(len(dict_treinamento['epoch'])):
                    f.write(f"{dict_treinamento['epoch'][i]};")
                    f.write(f"{dict_treinamento['loss total'][i]};")
                    f.write(f"{dict_treinamento['loss equação deslocamento'][i]};")
                    f.write(f"{dict_treinamento['loss equação pressão'][i]};")
                    f.write(f"{dict_treinamento['loss contorno direito'][i]};")
                    f.write(f"{dict_treinamento['loss contorno esquerdo'][i]};")
                    f.write(f"{dict_treinamento['loss condição inicial'][i]};")
                    f.write(f"{dict_treinamento['du'][i]};")
                    f.write(f"{dict_treinamento['dp'][i]}\n")
                f.close()
            dict_treinamento = {'epoch': [], 'loss total': [], 'loss equação deslocamento': [],
                                'loss equação pressão': [],
                                'loss contorno direito': [], 'loss contorno esquerdo': [], 'loss condição inicial': [],
                                'du': [], 'dp': []}
  
        if epoch == limite_epoch:
            model.save('model')
            break
  
        if cycle_counter == 2 * step_size:
            cycle_counter = 1
            factor *= 0.99999
            learning_rate = 10 ** (min_loglr + delta_loglr * factor)
        else:
            factor *= 0.99999
            learning_rate = 10 ** (min_loglr + abs(delta_loglr - var_loglr * cycle_counter) * factor)
            cycle_counter += 1
  
    pass


if __name__ == '__main__':
    main()
