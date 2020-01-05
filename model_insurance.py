import logging as logger
import tensorflow as tf
from tensorflow import keras

# define modelo da rede
logger.info("Iniciando modelo da rede...")
model = keras.Sequential()

# define inputs do modelo (definir com dataset de entrada)
inputs = 'dataset aqui'
targets = 'targets aqui'
# OU inicializa com inputs = keras.Input()

n1 = 'nº unidades da camada oculta'
n2 = 'nº unidades da camada de saída'

# define parametros iniciais do modelo (peso, epocas e taxa de aprendizado)
weights = tf.random_uniform_initializer(minval=-0.1, maxval=0.1) 
# é definido aleatoriamento pelo modelo se não for passado
bias = tf.random_uniform_initializer(minval=-0.1, maxval=0.1) 
learning_rate = 0.01
epochs = 100

# Para camada oculta:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?version=stable
#### calcular equação linear (nº de neuronios na camada = nº de linhas?)
#### calcular função de ativação
hidden = keras.layers.Dense(n1, 
                            activation = 'relu', 
                            kernel_initializer = weights,
                            bias_initializer = bias)
model.add(hidden)

# Para camada de saída:
    # calcular equação linear (nº de camada = nº de linhas?)
    # calcular função de ativação
    # calcular função de erro
output = keras.layers.Dense(n2, activation = 'sigmoid')
model.add(output)

# Executa otimização (backpropagation a partir do erro)
model.compile(optimizer='SGD', loss='MSE')

# Inicia o treinamento (epocas = iterações)
# https://www.tensorflow.org/api_docs/python/tf/keras/Model?version=stable#fit
model.fit(inputs, targets, epochs = epochs)
