"""#Utility functions"""
SEED = 42
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
def show_digit(x, y, pred=None):
  fig = plt.figure(figsize=(3, 2))
  plt.title('True: {label} and Predicted: {pred}'.format(label=np.argmax(y), pred=np.argmax(pred)))
  plt.imshow(x.reshape((28,28)), cmap='Greys_r')

def l_2_dist(orig_img, new_img):
    return np.sqrt(np.sum((orig_img.ravel()-new_img.ravel())**2))

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
  l_2 = round(l_2_dist(x_clean, x_adv), 4)
  l_inf = round(l_inf_dist(x_clean.ravel(), x_adv.ravel()), 4)
  ws = round(wasserstein_distance(x_clean.ravel(), x_adv.ravel()), 4)
  ssim_d = round(ssim(x_clean.ravel(), x_adv.ravel()), 4)
  psnr_d = round(psnr(x_clean.ravel(), x_adv.ravel()),4)
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

def browse_mis_samples(clean_images, adv_images, y_true, y_pred, verbose=True):
  total_images = len(adv_images)
  mis_preds = get_mis_preds(y_true, y_pred)


  #print("Before Clean shape: {} adv shape: {}".format
   #     (clean_images.shape, adv_images.shape))


  clean_images = clean_images[mis_preds]
  adv_images = adv_images[mis_preds]

  #print("Afer mispred Clean shape: {} adv shape: {}".format
   #     (clean_images.shape, adv_images.shape))


  #N = clean_images.shape[0]
  N = len(mis_preds)
  if (N == 0):
    print("There are zero mis-classified images!")
    return

  print("Mis-classified Images: {} out of Total: {}".format(N, total_images))
  if(verbose):
    print("Mis preds: ", mis_preds)
    print("Mean L2 Dist.: ", get_dataset_l_2_dist(clean_images, adv_images))
  rows = 1
  columns = 2
  def view_image(i=0):
      fig = plt.figure(figsize=(5, 3))
      true_label = np.argmax(y_true[mis_preds[i]])
      pred_label = np.argmax(y_pred[mis_preds[i]])
      fig.add_subplot(rows, columns, 1)
      plt.imshow(clean_images[i].reshape(28,28), cmap='Greys_r', interpolation='nearest')
      plt.axis('off')
      plt.title("#{} Clean Img.True:{}".format(i,true_label))

      fig.add_subplot(rows, columns, 2)
      plt.imshow(adv_images[i].reshape(28,28), cmap='Greys_r', interpolation='nearest')
      plt.axis('off')
      plt.title("Adv Img. Predicted: {} ".format(pred_label))
      dist = "L2 Dist: {}".format(l_2_dist(clean_images[i], adv_images[i]))
      fig.suptitle(dist, y=0.1)
  interact(view_image, i=(0, N-1))

def browse_all_samples(clean_images, adv_images, y_true, y_pred, mis= True):

  N = len(clean_images)
  print("Total  Images: ", N)
  rows = 1
  columns = 2
  def view_image(i=0):
      fig = plt.figure(figsize=(3, 2))
      true_label = np.argmax(y_true[i])
      pred_label = np.argmax(y_pred[i])
      fig.add_subplot(rows, columns, 1)
      plt.imshow(clean_images[i].reshape(28,28), cmap='Greys_r', interpolation='nearest')
      plt.axis('off')
      plt.title("#{} Clean Image. True Label: {}".format(i,true_label))

      fig.add_subplot(rows, columns, 2)
      plt.imshow(adv_images[i].reshape(28,28), cmap='Greys_r', interpolation='nearest')
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
  dataset_temp = copy.deepcopy(dataset)
  clean_y = dataset_temp.pop('CLEAN_Y')
  filename = basedir + "CLEAN_Y" + ".csv"
  print("Saving CLEAN_Y: ", filename)
  np.savetxt(filename, clean_y, delimiter=',' )
  for key in dataset_temp:
    print("Saving..: ", key)
    filename = basedir + key + ".csv"
    print(filename)
    x = np.reshape(dataset_temp[key], ((dataset_temp[key].shape[0],
                                        dataset_temp[key].shape[1] * dataset_temp[key].shape[2])))
    np.savetxt(filename, x, delimiter=',' )

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
  params = copy.deepcopy(eval_params)
  x_test_random = params.pop('CLEAN_X')
  y_test_random = params.pop('CLEAN_Y')

  print("\nEvaluating CLEAN: ")
  predictions = classifier.predict(x_test_random)
  classifier_evals['CLEAN'] = get_perf_metrics(y_test_random, predictions)

  for key in params:
    print("\nEvaluating: ", key)
    test_x = params[key]
    predictions = classifier.predict(test_x)
    classifier_evals[key] = get_perf_metrics(y_test_random, predictions)
  return classifier_evals
