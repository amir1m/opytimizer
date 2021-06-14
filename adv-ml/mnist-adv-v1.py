# %load_ext autoreload
# %autoreload 2


SEED = 42

import sys
import keras
sys.path.append('.')
sys.path.append('./adv-ml/')
import opytimizer.utils.logging as l
logger = l.get_logger(__name__)

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
import art.attacks.evasion as evasion
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
import opytimizer.optimizers.science
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
tf.compat.v1.disable_eager_execution()

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

dataset = 'cifar10'
#tf.compat.v1.disable_eager_execution()
# Read MNIST dataset
#(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str(dataset))

#model_logit = create_model()

#model_logit.fit(x_train, y_train, epochs=20, batch_size=128)

model_logit = load_model('adv-ml/models/'+dataset)
#classifier = KerasClassifier(model=model_logit)
#model_cifar = load_model('adv-ml/models/cifar-10', compile = False)

# n_samples = 1
# x_test_random, y_test_random, rand_ind = get_random_correct_samples(
# n_samples, x_test, y_test, model_logit.predict(x_test), seed = 0)
# logger.info("SEED: 0")

# n_samples = 2
# x_test_random, y_test_random, rand_ind = get_random_correct_samples(
# n_samples, x_test, y_test, model_logit.predict(x_test), seed = 0)

# x_test_random, y_test_random, rand_ind = get_random_correct_samples(
# n_samples, x_test, y_test, model_cifar.predict(x_test), seed = 0)



dim = x_test.shape
#dim
#y_test_random[0]
#show_image(x_test_random[0], np.argmax(y_test_random[0]))

#show_digit(x_test_random,y_test_random)
# from importlib import reload # reload
# reload(opytimizer.optimizers.misc)

# CIFAR
# loss, l_2_mean, query_mean, x_test_opyt = get_opyt_adv(model_logit,
#                                                      x_test_random,
#                                                      y_test_random,
#                                                      iterations=50,
#                                                      epsilon=.05,
#                                                      agents=25,
#                                                      max_l_2=2,
#                                                      l_2_mul=0.5,
#                                                      dim=dim
#                                                      )
## MNIST
# loss, l_2_mean, query_mean, x_test_opyt = get_opyt_adv(model_logit,
#                                                      x_test_random,
#                                                      y_test_random,
#                                                      iterations=60,
#                                                      epsilon=.1,
#                                                      agents=30,
#                                                      max_l_2=2,
#                                                      l_2_mul=0.5,
#                                                      dim=dim
#                                                      )

# y_target = np.array([5, 8, 1, 3, 6])
# loss, l_2_mean, query_mean, x_test_opyt = get_opyt_target_adv(model_logit,
#                                                      x_test_random,
#                                                      y_test_random,
#                                                      y_target=y_target,
#                                                      iterations=25,
#                                                      epsilon=1.1,
#                                                      agents=25,
#                                                      max_l_2=5,
#                                                      l_2_mul=4,
#                                                      )


# loss, l_2_mean, query_mean, x_test_opyt = get_nvg_adv(model_logit,
#                                                      x_test_random,
#                                                      y_test_random,
#                                                      iterations=3000,
#                                                      epsilon=.1,
#                                                      max_l_2=2,
#                                                      dim=dim
#                                                      )


# adv_dataset_soft = generate_adv_datsets(model_logit,x_test, y_test, n=2, seed=0,
#                                         attack_list=['FGSM', 'BOUNDARY',
#                                                      'SIMBA', 'HOPSKIPJUMP'])

n_samples = 10
adv_dataset_soft = generate_adv_datsets(model_logit,x_test, y_test, n=n_samples,
                                        attack_list=['FGSM','SIMBA','HOPSKIPJUMP','OPYT'], dim=
                                        (n_samples, dim[1], dim[2],dim[3]), seed=0)

#evals = evaluate_classifier(model_logit, adv_dataset_soft)

#logger.info(f'Evaluation Result: {evals}')

for key in adv_dataset_soft:
    if '_X' not in key and '_Y' not in key:
        logger.info(f'{key} : {adv_dataset_soft[key]}')

save_dataset(adv_dataset_soft,'soft/')




# np.savetxt('x_test_random_'+dataset+'.csv', x_test_random.reshape((dim[0], dim[1]*dim[2]*dim[3])), delimiter=',')
# np.savetxt('y_test_random_'+dataset+'.csv', y_test_random, delimiter=',')
#
# np.savetxt('x_test_opyt_'+dataset+'.csv', x_test_opyt.reshape((dim[0], dim[1]*dim[2]*dim[3])), delimiter=',')
# np.savetxt('y_pred_opyt_'+dataset+'.csv', model_logit.predict(x_test_opyt), delimiter=',')
