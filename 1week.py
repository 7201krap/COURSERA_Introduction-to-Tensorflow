'''

The LOSS function
-> measures the guessed answers against the known correct answers and measures how well or how badly it did.

The OPTIMIZER function to make another guess.
-> Based on how the loss function went, it will try to minimize the loss.

It will repeat this for the number of EPOCHS which you will see shortly.
-> But first, here's how we tell it to use 'MEAN SQUARED ERROR' for the LOSS and 'STOCHASTIC GRADIENT DESCENT' for the OPTIMIZER.

'''

import tensorflow as tf
import numpy as np
from tensorflow import keras

# 여기에서 Dense는 layer가 1개 있다는 것을 의미한다. Dense x개 -> layer x개
# Dense : a layer of connected neurons
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

'''
MODEL.FIT
-> The process of training the neural network, where it 'learns' the relationship between the Xs and Ys is in the model.fit call. It will do it for the number of epochs you specify.
'''

model.fit(xs, ys, epochs=500)

'''
MODEL.PREDICT
-> now you have a model that has been trained to learn the relationship between X and Y. You can use the model.predict method to have it figure out the Y for a previously unknown X
'''

print(model.predict([10.0]))

'''
convergence : The process of getting very close to the correct answer
'''
