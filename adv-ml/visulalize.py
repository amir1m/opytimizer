# %load_ext autoreload
# %autoreload 2

import numpy as np
SEED = 42

import sys
import keras
sys.path.append('.')


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

SEED = 42
np.random.seed(SEED)

from attack_utils import *

n_samples = 1

image = np.genfromtxt('x_test_opyt.csv', delimiter=',')
image1 = image.reshape((28,28,1))
show_digit(image1,1, model_logit.predict((image1.reshape((1,28,28,1)))))
