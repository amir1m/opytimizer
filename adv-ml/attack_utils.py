"""#Utility functions"""
import sys
sys.path.append('.')
sys.path.append('./adv-ml/')


import opytimizer.utils.logging as l
logger = l.get_logger(__name__)

SEED = 42
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from scipy.stats import wasserstein_distance
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions



from copy import deepcopy

def show_digit(x, y, pred=None):
  fig = plt.figure(figsize=(3, 2))
  plt.title('True: {label} and Predicted: {pred}'.format(label=np.argmax(y), pred=np.argmax(pred)))
  plt.imshow(x.reshape((28,28)), cmap='Greys_r')

cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
mnist_class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
def show_image(image, predictions):
  label = np.argmax(predictions)
  fig = plt.figure()
  fig.set_size_inches(5,2)
  plt.imshow(image, cmap=plt.cm.binary)
  plt.xlabel(cifar_class_names[label])
  plt.show()

def l_2_dist(orig_img, new_img):
    orig_img_c = np.copy(orig_img)
    new_img_c = np.copy(new_img)
    return np.sqrt(np.sum((orig_img_c.ravel()- new_img_c.ravel())**2))

def l_inf_dist(orig_img, new_img):
    return np.max(np.abs(orig_img.ravel() - new_img.ravel()))

def l_0_dist(orig_img, new_img):
    return np.sum((orig_img.ravel() - new_img.ravel()) != 0)

def get_dataset_l_2_dist(orig_img, new_img):
  dist = []
  for i in range(orig_img.shape[0]):
    dist.append(np.sqrt(np.sum((orig_img[i].ravel()-new_img[i].ravel())**2)))
  return np.mean(dist)

def get_all_dist(x_clean, x_adv):
  #l_2 = round(l_2_dist(x_clean, x_adv), 4)
  l_2 = l_2_dist(x_clean, x_adv)
  logger.info(f'L_2:{l_2}')
  l_inf = round(l_inf_dist(x_clean.ravel(), x_adv.ravel()), 4)
  ws = round(wasserstein_distance(x_clean.ravel(), x_adv.ravel()), 4)
  ssim_d = round(ssim(x_clean.ravel(), x_adv.ravel()), 4)
  psnr_d = round(psnr(x_clean.ravel().clip(0,1), x_adv.ravel().clip(0,1)),4)
  dist =  {'L2': l_2, 'L-INF': l_inf,'WS': ws, 'ssim': ssim_d, 'psnr': psnr_d}
  return dist

def get_dist_metrics(dist_params):
  dist_metrics = {}
  for key in dist_params:
    #print('\nDistances for ', key)
    x_clean = dist_params[key][0]
    x_adv = dist_params[key][1]
    try:
      if(x_clean.shape != x_adv.shape):
        raise ValueError("get_dist_metrics:Size of the both datsets not same for: ", key)
    except ValueError as ve:
      print(ve)
      continue
    dist_metrics[key] = {'L2': round(get_dataset_l_2_dist(x_clean, x_adv), 4),
                         'L-INF': round(l_inf_dist(x_clean.ravel(), x_adv.ravel()), 4),
                         'WS': round(wasserstein_distance(x_clean.ravel(), x_adv.ravel()), 4),
                         'ssim': round(ssim(x_clean.ravel(), x_adv.ravel()), 4),
                         'psnr': round(psnr(x_clean.ravel(), x_adv.ravel()),4)
                         }
    #print(dist_metrics[key])
  return dist_metrics

### USAGE
# dist_params = {'FGSM': [x_test_random, x_test_random_fgsm],
#                'ZOO' : [x_test_random, x_test_zoo],
#                'BOUNDARY': [x_test_random, x_test_random_boundary],
#                'CSO' : [x_test_random, x_test_cso]

# }

def counter(func):
  def wrapper(*args, **kwargs):
    wrapper.count += 1
    # Call the function being decorated and return the result
    return func(*args, **kwargs)
  wrapper.count = 0
  # Return the new decorated function
  return wrapper

def get_accuracy(y_pred, y_true):
  acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)) / y_true.shape[0]
  #print("\nTest accuracy : %.2f%%" % (acc * 100))
  return acc * 100

def get_mis_preds(y_true, y_preds):
  return np.where(np.argmax(y_true, axis=1) != np.argmax(y_preds, axis=1))[0]

def get_correct_preds(y_true, y_preds):
  return np.where(np.argmax(y_true, axis=1) == np.argmax(y_preds, axis=1))[0]

def browse_mis_samples(clean_images, adv_images, y_true, y_pred, dim=(1,28,28,1),class_name=mnist_class_names, verbose=True):
  total_images = len(adv_images)
  mis_preds = get_mis_preds(y_true, y_pred)
  # print("MIS PREDS: ", mis_preds)
  # print("Y True array:", y_true)

  clean_images = clean_images[mis_preds]
  adv_images = adv_images[mis_preds]

  N = len(mis_preds)
  if (N == 0):
    print("There are zero mis-classified images!")
    return

  print("CORRECTED Mis-classified Images: {} out of Total: {}".format(N, total_images))
  if(verbose):
    print("Mis preds: ", mis_preds)
    print("Mean L2 Dist.: ", get_dataset_l_2_dist(clean_images, adv_images))
  rows = 1
  columns = 2
  def view_image(i=0):
      fig = plt.figure(figsize=(5, 3))
      true_label = class_name[np.argmax(y_true[mis_preds[i]])]
      pred_label = class_name[np.argmax(y_pred[mis_preds[i]])]
      fig.add_subplot(rows, columns, 1)
      plt.imshow(clean_images[i].reshape(dim[1], dim[2], dim[3]), cmap='Greys_r', interpolation='nearest')
      plt.axis('off')
      plt.title("#{} True:{}".format(i,true_label))

      fig.add_subplot(rows, columns, 2)
      plt.imshow(adv_images[i].reshape(dim[1], dim[2], dim[3]), cmap='Greys_r', interpolation='nearest')
      plt.axis('off')
      plt.title("Predicted: {} ".format(pred_label))
      dist = "L2 Dist: {}".format(l_2_dist(clean_images[i], adv_images[i]))
      fig.suptitle(dist, y=0.1)
  interact(view_image, i=(0, N-1))

def browse_all_samples(clean_images, adv_images, y_true, y_pred, mis= True):

  N = len(clean_images)
  print("Total  Images: ", N)
  print("True label array:", y_true)
  rows = 1
  columns = 2
  def view_image(i=0):
      fig = plt.figure(figsize=(3, 2))
      true_label = np.argmax(y_true[i])
      pred_label = np.argmax(y_pred[i])
      fig.add_subplot(rows, columns, 1)
      plt.imshow(clean_images[i].reshape(dim[1], dim[2]), cmap='Greys_r', interpolation='nearest')
      plt.axis('off')
      plt.title("#{} Clean Image. True Label: {}".format(i,true_label))

      fig.add_subplot(rows, columns, 2)
      plt.imshow(adv_images[i].reshape(dim[1], dim[2]), cmap='Greys_r', interpolation='nearest')
      plt.axis('off')
      plt.title("Adv Image. Predicted: {} ".format(pred_label))
  interact(view_image, i=(0, N-1))

def get_random_correct_samples(size,x_test, y_true, y_pred, seed = SEED):
  np.random.seed(seed)
  #y_test_label = np.argmax(y_test, axis=1)
  #y_pred_label = np.argmax(y_pred, axis=1)
  #print("y_test_label", y_test_label)

  correct_indices = get_correct_preds(y_true, y_pred)
  rand_indices = np.random.choice(correct_indices, size = size)
  return x_test[rand_indices], y_true[rand_indices], rand_indices

def get_random_any_samples(size,x_train, y_train):
  np.random.seed(SEED)
  #y_test_label = np.argmax(y_test, axis=1)
  #y_pred_label = np.argmax(y_pred, axis=1)
  #print("y_test_label", y_test_label)
  rand_indices = np.random.choice(np.array(y_train.shape[0]), size = size)
  return x_train[rand_indices], y_train[rand_indices], rand_indices

def save_dataset(dataset, basedir):
  dataset_temp = deepcopy(dataset)
  for key in dataset_temp:
      if '_X' in key:
          adv_x = dataset_temp[key]
          filename = basedir + key + ".csv"
          logger.info(f'Saving : {key}, filename:{filename}')
          x = np.reshape(adv_x, ((adv_x.shape[0],
                                              adv_x.shape[1] * adv_x.shape[2] * adv_x.shape[3])))
          np.savetxt(filename, x, delimiter=',' )
      elif '_Y' in key:
        adv_y = dataset_temp[key]
        filename = basedir + key + ".csv"
        logger.info(f"Saving: {key}, filename:{filename}")
        np.savetxt(filename, adv_y, delimiter=',' )

def load_adv_dataset(params, basedir, x_dim):
  print("Loading dataset from dir: ", basedir)
  dataset = {}
  clean_y = params.pop(0)
  filename = basedir + "CLEAN_Y" + ".csv"
  print("Loading CLEAN_Y: ", filename)
  dataset['CLEAN_Y'] = np.genfromtxt(filename,delimiter=',' )
  for key in params:
    print("Loading..: ", key)
    filename = basedir + key + ".csv"
    print(filename)
    x_temp = np.genfromtxt(filename, delimiter = ',')
    dataset[key] = np.reshape(x_temp, (x_dim))
  return dataset

"""## Evaluation Functions"""

def get_perf_metrics(actual, predictions, verbose = 1):
  accuracy = accuracy_score(np.argmax(actual, axis=1), np.argmax(predictions,
                                                                 axis=1))
  precision = precision_score(np.argmax(actual, axis=1), np.argmax(predictions,
                                                                   axis=1), average='micro')
  recall = recall_score(np.argmax(actual, axis=1), np.argmax(predictions,
                                                                   axis=1), average='micro')
  f1 = f1_score(np.argmax(actual, axis=1), np.argmax(predictions,
                                                                   axis=1), average='micro')

  if (verbose != 0):
    print("Accuracy: %f, Precision: %f, \nRecall: %f, F1 Score: %f\n"%
          (accuracy, precision, recall, f1))
  return {
      "accuracy" : accuracy,
      "precision" : precision,
      "recall" : recall,
      "f1_score" : f1
  }

def evaluate_classifier(classifier, eval_params):
  classifier_evals = {}
  params = deepcopy(eval_params)
  x_test_random = params.pop('CLEAN_X')
  y_test_random = params.pop('CLEAN_Y')

  dataset_x = ['_X' in eval_params.keys ]
  dataset_y = ['_Y' in eval_params.keys]
  logger.info(f"dataset_x: {dataset_x}")
  logger.info(f"dataset_y: {dataset_y}")


  logger.info("Evaluating CLEAN: ")
  predictions = classifier.predict(x_test_random)
  classifier_evals['CLEAN'] = { 'accuracy': accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(predictions,
                                                                 axis=1))}

  for x in dataset_x:
    logger.info(f'Evaluating:{x}')
    adv_x = params.pop(key)
    adv_y = params.pop(key.replace('X', 'Y'))
    mis_preds = get_mis_preds(y_test_random, adv_y)
    #print("MIS PREDS: ", mis_preds)
    #print("Y True array:", y_true)
    clean_images = x_test_random[mis_preds]
    adv_images = adv_x[mis_preds]
    classifier_evals[key] = { 'accuracy': accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(adv_y,
                                                                   axis=1)),
                                                                   'dist':get_all_dist(clean_images,adv_x)}

  return classifier_evals

def get_imagenet_top_1_pred(model, x):
  x_p = np.copy(x)
  preds = model.predict(preprocess_input(x_p))
  top_1 = decode_predictions(preds, top=1)[0]
  top_1_hard_label = top_1[0][1]
  return top_1_hard_label

def get_imagenet_true_label(img_path):
  true_label = img_path.split('_')[1:]
  true_label = "_".join(true_label)
  true_label = true_label.split('.')[0]
  return true_label

def load_imagenet_image(img_path, input_shape):
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  #x = preprocess_input(x)
  return x
