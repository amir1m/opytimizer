# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('.')
sys.path.append('./adv-ml/')
import numpy as np
SEED = 42

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact
from tensorflow.keras.preprocessing import image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

SEED = 42
np.random.seed(SEED)

from attack_utils import *

#n_samples = 100

# image = np.genfromtxt('x_test_opyt.csv', delimiter=',')
# image1 = image.reshape((n_samples, 28,28,1))
# show_digit(image1[0],1, model_logit.predict((image1[0].reshape((1,28,28,1)))))


n_samples = 1
images = np.genfromtxt('x_test_random_mnist.csv', delimiter=',')
orig_images = images.reshape((n_samples, 28,28,1))
orig_labels = np.genfromtxt('y_test_random.csv', delimiter=',')

images = np.genfromtxt('x_test_opyt_mnist.csv', delimiter=',')
adv_images = images.reshape((n_samples, 28,28,1))
adv_labels = np.genfromtxt('y_pred_opyt_mnist.csv', delimiter=',')

#np.argmax(adv_labels)
#np.savetxt('y_pred_opyt.csv', model_logit.predict(adv_images), delimiter=',')

l_2_dist(orig_images, adv_images)
browse_mis_samples(orig_images, adv_images,orig_labels, adv_labels)

i=4
browse_mis_samples(orig_images[i].reshape((1,28,28,1)), adv_images[i].reshape((1,28,28,1)),orig_labels[i], adv_labels[i])


i=0
show_digit(adv_images[i],orig_labels[i], adv_labels[i])

i=0
show_digit(orig_images[i],orig_labels[i], orig_labels[i])


n_samples = 100
dim = (n_samples,32,32,3)
images = np.genfromtxt('x_test_random_cifar10.csv', delimiter=',')
orig_images = images.reshape(dim)
orig_labels = np.genfromtxt('y_test_random_cifar10.csv', delimiter=',')

images = np.genfromtxt('x_test_opyt_cifar10.csv', delimiter=',')
adv_images = images.reshape(dim)
adv_labels = np.genfromtxt('y_pred_opyt_cifar10.csv', delimiter=',')
l_2_dist(orig_images, adv_images)
show_image(adv_images[0], adv_labels[0])


show_image(orig_images[0], orig_labels[0])



#==================IMAGENET================#
n_samples = 1
dim = (224,224,3)
images = np.genfromtxt('x_clean_imagenet.csv', delimiter=',')
orig_images = image.array_to_img(images.reshape((224,224,3)))
#orig_images_pr = load_imagenet_image('x_clean_imagenet.csv', (224,224))
plt.imshow(orig_images)


images_adv = np.genfromtxt('x_adv_imagenet.csv', delimiter=',')
adv_images = image.array_to_img(images_adv.reshape((224,224,3)))
plt.imshow(adv_images)
