"""## Evaluation Functions"""

import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix


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
