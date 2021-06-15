import sys
sys.path.append('.')
sys.path.append('./adv-ml/')
import opytimizer.utils.logging as l
logger = l.get_logger(__name__)

import keras.backend as K
import tensorflow as tf
tf.__version__
#tf.compat.v1.disable_eager_execution()
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential, load_model
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models
import numpy as np
SEED = 42
np.random.seed(SEED)
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score, precision_score, recall_score

from art.attacks.evasion import FastGradientMethod, ZooAttack, CarliniL2Method, BoundaryAttack, SimBA, HopSkipJump
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset

from copy import deepcopy
from attack_utils import *

dataset = 'cifar10'
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str(dataset))
cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


model_logit = load_model('adv-ml/models/'+dataset)

show_image(x_test[0],1)

y_test[0]

y_pred = model_logit.predict(x_test[0].reshape(1,32,32,3))[0]
y_pred

cifar_class_names[np.argmax(y_pred)]

sorted_pred = np.argsort(y_pred)
sorted_pred

target = deepcopy(y_pred)
sorted_pred[-2]
target[sorted_pred[-1]] = y_pred[sorted_pred[-2]]
target[sorted_pred[-2]] = y_pred[sorted_pred[-1]]
target

def adv_loss(clean, adv):
  # clean_pred = model_logit(K.reshape(clean, (1,32,32,3)))
  # x_adv = K.reshape(adv, (1,32,32,3))
  # adv_pred = model_logit(x_adv)
  return K.square(clean - K.reshape(adv, (32,32,3)))
  # sorted_adv_pred = tf.argsort(adv_pred)
  #
  # loss = K.square(adv_pred[sorted_y_pred[9]] - adv_pred[sorted_y_pred[8]])
  # logger.info(f'loss:{loss}')
  # return loss


latent_dim = 64
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(32*32*3, activation='sigmoid'),
      layers.Reshape((32, 32,3))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
