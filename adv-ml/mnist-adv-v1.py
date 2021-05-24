SEED = 42

import sys
import keras
sys.path.append('.')
sys.path.append('./adv-ml/')

from keras.models import Sequential
from tensorflow.keras.models import  load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
import numpy as np
np.random.seed(SEED)
from numpy import vstack
from numpy import hstack
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score, precision_score, recall_score

from art.attacks.evasion import FastGradientMethod, CarliniL2Method, ZooAttack, CarliniL2Method, BoundaryAttack, SimBA, HopSkipJump
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset

import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
#from ipywidgets import interact

import opytimizer
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.functions import ConstrainedFunction
from opytimizer.optimizers.swarm import CS
from opytimizer.optimizers.evolutionary import GA, HS, FOA, GP, DE, IWO, EP, ES, CRO
from opytimizer.optimizers.misc.aoa import AOA
from opytimizer.optimizers.misc.hc import HC
from opytimizer.optimizers.misc.cem import CEM
from opytimizer.spaces import SearchSpace, TreeSpace

import nevergrad as ng

from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import copy

import matplotlib.pyplot as plt
#import pandas as pd
#%matplotlib inline
#from ipywidgets import interact
from datetime import datetime
import os

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

SEED = 42
np.random.seed(SEED)

import attack_utils

import logging
# Gathers all instantiated loggers, even the children
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# Iterates through all loggers and set their level to INFO
for logger in loggers:
    logger.setLevel(logging.INFO)


tf.__version__
import keras
keras.__version__
# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))

show_digit(x_test[0],1,1)

#model_logit = create_model()
#classifier = KerasClassifier(model=model_logit)
#model_logit.fit(x_train, y_train, epochs=20, batch_size=128)

model_logit = load_model('adv-ml/models/mnist', compile = False)
#
# with open('/Volumes/macos1/opytimizer/adv-ml/models/mnist_model_config.json') as json_file:
#     json_config = json_file.read()
# model_logit = tf.keras.models.model_from_json(json_config)

n_samples = 1
x_test_random, y_test_random, rand_ind = get_random_correct_samples(n_samples, x_test, y_test, model_logit.predict(x_test), seed = 1231)

show_digit(x_test_random[0],1)

loss, l_2_mean, query_mean, x_test_opyt = get_opyt_adv(
model_logit, x_test_random,y_test_random,
iterations=100, epsilon=.35, max_l_2=2, agents=10
)


show_digit(x_test_random[0],1)
show_digit(x_test_random[0]/100.0, y_test_random[0],
model_logit(x_test_random[0].reshape((1,28,28,1))/100.0))

l_2_dist(x_test_random[0], x_test_random[0]/100)
show_digit(x_test_random,1)

np.max(x_test_opyt.ravel())
print(x_test_random.ravel()
