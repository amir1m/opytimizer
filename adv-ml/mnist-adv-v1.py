# %load_ext autoreload
# %autoreload 2


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

from attack_utils import *

from attack import *
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

#model_logit = create_model()
#classifier = KerasClassifier(model=model_logit)
#model_logit.fit(x_train, y_train, epochs=20, batch_size=128)

model_logit = load_model('adv-ml/models/mnist', compile = False)

n_samples = 1
x_test_random, y_test_random, rand_ind = get_random_correct_samples(
n_samples, x_test, y_test, model_logit.predict(x_test), seed = 42)

# from importlib import reload # reload
# reload(opytimizer.optimizers.misc)


loss, l_2_mean, query_mean, x_test_opyt = get_opyt_adv(model_logit,
                                                     x_test_random,
                                                     y_test_random,
                                                     iterations=50,
                                                     epsilon=1.05,
                                                     max_l_2=3,
                                                     agents = 15
                                                     )

np.savetxt('x_test_opyt.csv', x_test_opyt.reshape((n_samples, 784)), delimiter=',')

show_digit(x_test_random[0],1)
