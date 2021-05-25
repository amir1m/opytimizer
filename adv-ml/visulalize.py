# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('.')
sys.path.append('./adv-ml/')
import numpy as np
SEED = 42



from ipywidgets import interact
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

SEED = 42
np.random.seed(SEED)

from attack_utils import *

n_samples = 10

# image = np.genfromtxt('x_test_opyt.csv', delimiter=',')
# image1 = image.reshape((n_samples, 28,28,1))
# show_digit(image1[0],1, model_logit.predict((image1[0].reshape((1,28,28,1)))))



images = np.genfromtxt('x_test_random.csv', delimiter=',')
orig_images = images.reshape((n_samples, 28,28,1))
orig_labels = np.genfromtxt('y_test_random.csv', delimiter=',')

images = np.genfromtxt('x_test_opyt.csv', delimiter=',')
adv_images = images.reshape((n_samples, 28,28,1))
adv_labels = np.genfromtxt('y_pred_opyt.csv', delimiter=',')

#np.savetxt('y_pred_opyt.csv', model_logit.predict(adv_images), delimiter=',')

browse_mis_samples(orig_images, adv_images,orig_labels, adv_labels)

i=8
show_digit(adv_images[i],orig_labels[i], adv_labels[i])

i=8
show_digit(orig_images[i],orig_labels[i], adv_labels[i])
