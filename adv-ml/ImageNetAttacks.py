# TensorFlow ResNet50
import sys
import keras
sys.path.append('.')
sys.path.append('./adv-ml/')
import opytimizer
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.functions import ConstrainedFunction
from opytimizer.optimizers.swarm import CS
from opytimizer.optimizers.evolutionary import GA, HS, FOA, GP, DE, IWO, EP, ES
from opytimizer.optimizers.misc.aoa import AOA
from opytimizer.optimizers.misc.hc import HC
from opytimizer.optimizers.misc.cem import CEM
import opytimizer.optimizers.science
from opytimizer.spaces import SearchSpace, TreeSpace
import opytimizer
from opytimizer import Opytimizer
from opytimizer import Opytimizer
import opytimizer.utils.logging as l
logger = l.get_logger(__name__)
import logging
# Gathers all instantiated loggers, even the children
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# Iterates through all loggers and set their level to INFO
for logger in loggers:
    logger.setLevel(logging.INFO)


import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


from attack_utils import *
from attack import *
dataset = 'imagenet'
model = ResNet50(weights='imagenet')
input_shape = (224,224) # For ResNet50

img_path = './data/imagenet-sample-images/n01484850_great_white_shark.JPEG'

# img = image.load_img(img_path, target_size=(224, 224))
#
# plt.imshow(img)
#
# x = image.img_to_array(img)
# plt.imshow(x/255)
# x = np.expand_dims(x, axis=0) + 0.01
# x = preprocess_input(x)
#
# get_imagenet_top_1_pred(model,x)
#
x_clean = load_imagenet_image(img_path, input_shape)
y_clean = get_imagenet_true_label(img_path)
# #
# x_clean.shape
#plt.imshow(x_clean[0]/255)

n_samples = 1
dim = x_clean.shape
dim
logger.info(f'Dataset:{dataset}, dim:{dim}, y_clean:{y_clean}')
# x_adv, count, dist = get_adv_opyt_imagenet_example(model,
#                                           x_clean,
#                                           y_clean,
#                                           epsilon = .3,
#                                           iterations = 5,
#                                           max_l_2 = 2,
#                                           agents = 3,
#                                           l_2_mul=0.5,
#                                           dim=(1, 224, 224, 3)
#                                           )

x_adv, count, dist = get_adv_opyt_target_imagenet_example(model,
                                        x_clean,
                                        y_clean,
                                        #epsilon = 0.5,
                                        iterations=10,
                                        max_l_2=6,
                                        agents=3,
                                        l_2_mul=.5,
                                        dim=(1, 224, 224, 3)
                                        )

np.savetxt('x_clean_'+dataset+'.csv', x_clean.reshape((n_samples, dim[1]*dim[2]*dim[3])), delimiter=',')
np.savetxt('y_clean_'+dataset+'.csv', ["%s" % y_clean], fmt='%s')

np.savetxt('x_adv_'+dataset+'.csv', x_adv.reshape((n_samples, dim[1]*dim[2]*dim[3])), delimiter=',')
np.savetxt('y_adv_'+dataset+'.csv', ["%s" % get_imagenet_top_1_pred(model, x_adv)], fmt='%s')
