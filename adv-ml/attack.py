"""# Attack Function

## Driver Functions
"""
SEED =42
import opytimizer.utils.logging as l
logger = l.get_logger(__name__)
import os
import sys
#sys.path.append('.')
sys.path.append('/Users/amirmukeri/Projects/opytimizer/nevergrad/')
import numpy as np
import nevergrad as ng
import SwarmPackagePy
from copy import deepcopy
from attack_utils import *
#from tlbo import TLBO, helper_n_generations
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance, levy
import opytimizer
from opytimizer.spaces import SearchSpace
from opytimizer.functions import ConstrainedFunction
from opytimizer.core import Function
from opytimizer import Opytimizer


from art.attacks.evasion import FastGradientMethod, CarliniL2Method, ZooAttack, CarliniL2Method, BoundaryAttack, SimBA, HopSkipJump
import art.attacks.evasion as evasion
from art.estimators.classification import KerasClassifier

def counter(func):
  def wrapper(*args, **kwargs):
    wrapper.count += 1
    # Call the function being decorated and return the result
    return func(*args, **kwargs)
  wrapper.count = 0
  # Return the new decorated function
  return wrapper


## WIP
def generate_adv_datsets(model, x_test, y_test, attack_list,
                         n=10, epsilon=0.001, dim=(10, 28,28,1), seed=SEED):
  #np.random.seed(SEED)
  x_adv = {}
  classifier = KerasClassifier(model) #For ART attacks

  x_test_random, y_test_random, rand_indices = get_random_correct_samples(
      n, x_test, y_test, model.predict(x_test), seed=seed)
  logger.info(f'x_test_random shape:{x_test_random.shape} and y_test_random shape:{y_test_random.shape  }')
  x_adv['CLEAN_X'] = x_test_random
  x_adv['CLEAN_Y'] = y_test_random

  for attack in attack_list:
    if(attack == 'FGSM'):
      logger.info("Generating adv examples using attack FGSM")
      epsilon = 0.1  # Maximum perturbation
      adv_crafter = FastGradientMethod(classifier, eps=epsilon)
      x_adv[attack+'_X'] = adv_crafter.generate(x=x_test_random)
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      logger.info(f'Misclassified images by attack {attack}: {mis_preds}')
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])

    if(attack == 'CWL2'):
      logger.info("Generating adv examples using attack CWL2")
      #epsilon = 0.1  # Maximum perturbation
      adv_crafter = CarliniL2Method(classifier)
      x_adv[attack+'_X'] = adv_crafter.generate(x=x_test_random)
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])

    elif(attack == 'BOUNDARY'):
      logger.info("Generating adv examples using attack BOUNDARY")
      boundary = BoundaryAttack(classifier, targeted=False)
      x_adv[attack+'_X'] = boundary.generate(x_test_random)
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])

    elif(attack == 'ZOO'):
      logger.info("Generating adv examples using attack ZOO")
      zoo = ZooAttack(
          classifier=classifier,
          confidence=0.0,
          targeted=False,
          learning_rate=1e-2,
          max_iter=200,
          binary_search_steps=10,
          initial_const=1e-3,
          abort_early=True,
          use_resize=False,
          use_importance=False,
          nb_parallel=128,
          batch_size=1,
          variable_h=0.01
      )
      x_adv[attack+'_X'] = zoo.generate(x_test_random)
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])

    elif(attack == 'SIMBA'):
      logger.info("Generating adv examples using attack SIMBA")
      simba = SimBA(classifier)
      x_adv[attack+'_X'] = simba.generate(x_test_random)
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])

    elif(attack == 'HOPSKIPJUMP'):
      logger.info("Generating adv examples using attack HOPSKIPJUMP")
      hopskipjump = HopSkipJump(classifier)
      x_adv[attack+'_X'] = hopskipjump.generate(x_test_random)
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])

    elif(attack == 'OPYT'):
      logger.info("Generating adv examples using attack OPYT")
      #Already tuned hyper-parameters
      loss, l_2_mean, query_mean, x_test_opyt = get_opyt_adv(model,
                                                           x_test_random,
                                                           y_test_random,
                                                           iterations=60,
                                                           epsilon=1,
                                                           agents=30,
                                                           max_l_2=2,
                                                           l_2_mul=0.5,
                                                           dim=dim
                                                           )
      x_adv[attack+'_X'] = x_test_opyt
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])

    elif(attack == 'OPYT_TARGET'):
      logger.info("Generating Targetted adv examples using attack OPYT")
      #Already tuned hyper-parameters
      loss, l_2_mean, query_mean, x_test_opyt = get_opyt_target_adv(model,
                                                           x_test_random,
                                                           y_test_random,
                                                           iterations=20,
                                                           epsilon=1,
                                                           agents=20,
                                                           max_l_2=3,
                                                           l_2_mul=0.5,
                                                           dim=dim
                                                           )
      x_adv[attack+'_X'] = x_test_opyt
      x_adv[attack+'_Y'] = model.predict(x_adv[attack+'_X'])
      x_adv[attack+'_accu'] = accuracy_score(np.argmax(y_test_random, axis=1), np.argmax(x_adv[attack+'_Y'],axis=1))
      mis_preds = get_mis_preds(y_test_random, x_adv[attack+'_Y'])
      if (len(mis_preds) == 0):
        logger.info(f'Attack {attack} was not successful on any input images')
        x_adv[attack+'_dist'] = {}
        return x_adv
      x_adv[attack+'_dist'] =  get_all_dist(x_test_random[mis_preds], x_adv[attack+'_X'][mis_preds])


  return x_adv

### USAGE:
# %%time
# adv_dataset_soft = generate_adv_datsets(model_soft,x_test, y_test,
#                                         attack_list=['FGSM', 'BOUNDARY', 'ZOO',
#                                                      'SIMBA', 'HOPSKIPJUMP',
#                                                      'CSO'])
# x_test_soft_random = adv_dataset_soft['CLEAN_X']
# y_test_soft_random = adv_dataset_soft['CLEAN_Y']

#save_dataset(adv_dataset_soft,'/content/drive/MyDrive/adv-ml/soft/')



"""## Functions for Adv Example generation using CSO"""

def process_digit(x_clean, x_prop, epsilon, dim=(1,28,28,1)):
  # logger.info(f"X CLEAN SHAPE:{x_clean.shape}")
  # logger.info(f"X PROP SHAPE:{x_prop.shape}")
  x_clean_ravel = np.copy(x_clean.ravel())
  x_clean_ravel += x_prop * epsilon
  #x_clean_ravel = (x_clean_ravel-min(x_clean_ravel)) / (max(x_clean_ravel)-min(x_clean_ravel))
  x_clean_ravel = x_clean_ravel.clip(0,1)
  return x_clean_ravel.reshape(dim)

def process_imagenet(x_clean, x_prop, epsilon, dim=(1,28,28,1)):
  # logger.info(f"X CLEAN SHAPE:{x_clean.shape}")
  # logger.info(f"X PROP SHAPE:{x_prop.shape}")
  x_clean_ravel = np.copy(x_clean.ravel())
  x_clean_ravel += x_prop
  #x_clean_ravel = (x_clean_ravel-min(x_clean_ravel)) / (max(x_clean_ravel)-min(x_clean_ravel))
  x_clean_ravel = x_clean_ravel.clip(0,255)
  return x_clean_ravel.reshape((1,224,224,3))

def process_image_target(x_clean, x_prop, x_target, epsilon, dim=(1,28,28,1)):
  # logger.info(f"X CLEAN SHAPE:{x_clean.shape}")
  # logger.info(f"X PROP SHAPE:{x_prop.shape}")
  x_clean_ravel = np.copy(x_clean.ravel())
  x_prop = np.where(x_prop >= 0.9, 1, x_prop)
  x_prop = np.where(x_prop < 0.9, 0, x_prop)
  x_clean_ravel =  x_clean_ravel - x_prop
  #x_clean_ravel = (x_clean_ravel-min(x_clean_ravel)) / (max(x_clean_ravel)-min(x_clean_ravel))
  x_clean_ravel = x_clean_ravel.clip(0,1)
  return x_clean_ravel.reshape(dim)

def process_image_target_imagenet(x_clean, x_prop,epsilon, dim=(1,28,28,1)):
  # logger.info(f"X CLEAN SHAPE:{x_clean.shape}")
  # logger.info(f"X PROP SHAPE:{x_prop.shape}")
  x_clean_ravel = np.copy(x_clean.ravel())
  x_prop = np.where(x_prop >= .1, 0, x_prop)
  #x_prop = np.where(x_prop < 150, 0, x_prop)
  x_clean_ravel =  x_clean_ravel + x_clean_ravel * x_prop * .5
  #x_clean_ravel = (x_clean_ravel-min(x_clean_ravel)) / (max(x_clean_ravel)-min(x_clean_ravel))
  x_clean_ravel = x_clean_ravel.clip(0,255)
  return x_clean_ravel.reshape(dim)

# TAKES ARGUMENTS
def get_cso_adv(model, x_test_random, y_test_random,
                n=150, iterations = 1, pa=0.5, nest=784, epsilon = 3.55, max_l_2=4):
  iteration = round(iterations)
  n = round(n)
  logger.info("n: {} Iteration:{} and espilon: {}".format(n,iteration, epsilon))

  no_samples = len(x_test_random)
  adv_cso = np.empty((no_samples,28,28,1))

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    print("Generating example: ", i)
    adv_cso[i], count, dist = get_adv_cso_example(model, x_test_random[i], y_test_random[i] ,
                                        pa=pa, n=n, nest=nest, iterations = iteration,
                                        epsilon = epsilon)
    query_count.append(count)
    l_2.append(dist)
  x_test_cso = adv_cso
  y_pred_cso = model.predict(x_test_cso)
  acc = get_accuracy(y_pred_cso, y_test_random)
  #l_2 = get_dataset_l_2_dist(x_test_random, x_test_cso)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  print("\nTotal Examples: {}, Iterations:{}, espilon: {} and Max-L2:{} nests: {}\nAccuracy: {} Mean L2 Counted: {} Query: {}".format(
      len(y_test_random), iterations, epsilon, max_l_2,nest, acc, l_2_mean,query_mean,
      l_2_dist(x_test_random.ravel(), x_test_cso.ravel())))

  print("Accuracy: {} Mean L2 Counted: {} Query: {}".format(
      acc, l_2_mean,query_mean, l_2_dist(x_test_random.ravel(), x_test_cso.ravel())))

  #PRODUCTION
  if (acc == 0):
    return -l_2_mean, l_2_mean, query_mean, x_test_cso
  return  acc * (-l_2_mean),l_2_mean, query_mean, x_test_cso

  # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return l_2_mean
  # return  acc * l_2_mean

def get_adv_cso_example(model, x_clean, y_clean, n=100, pa=0.5, nest=784, iterations = 10, epsilon = 0.001):
  @counter
  def evaluate_acc(x):
    x_adv = process_digit(x_clean, x, epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    return predictions[np.argmax(y_clean)]
    #return l_2_dist(x_clean, x_adv)
    #return l_2_dist(x_clean, x_adv)
    #return 100

  total_indices = len(x_clean.ravel())
  lb = np.empty(total_indices)
  lb.fill(0)
  ub = np.empty(total_indices)
  ub.fill(1)
  alh = SwarmPackagePy.cso(n, evaluate_acc, lb, ub, total_indices, pa=pa,
                           nest=nest,
                           iteration=iterations)
  x = np.array(alh.get_Gbest())
  x_adv = process_digit(x_clean, x, epsilon)
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
  attack_succ = np.argmax(y_clean) != adv_pred
  print("Attack result:{}, Dist:{}".format(attack_succ, dist))
  return x_adv, evaluate_acc.count, dist

def get_adv_opyt_example(model, x_clean, y_clean,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20, l_2_mul=.5,
                        dim=(1,28,28,1)):
  eval_count = 0
  x_adv = None
  l2_iter = round(iterations)

  def evaluate_acc(x):
    nonlocal eval_count
    eval_count += 1
    x_adv = process_digit(x_clean, x.ravel(), epsilon, dim=dim)
    predictions = model.predict(x_adv.reshape(dim))[0]
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    dist = float(l_2_dist(x_clean, x_adv))
    #dist = np.exp(-(l_2_dist(x_clean, x_adv)**2)/2)
    if(result != actual):
      # if (dist > max_l_2):
      #   return float(dist) * 10
      # else:
      #   return float(dist)
      return float(dist)
    else:
      #predictions.sort()
      #return float(10*(predictions[-1] - predictions[-2]) + 10 * dist)
      #return float(10*np.amax(predictions) + 10)
      return 100

  def l_2_constraint(x):
    return l_2_dist(x_clean, x) < max_l_2

  @counter
  def inequality_constraint(x):
    x_adv = process_digit(x_clean, x.ravel(), epsilon, dim=dim)
    predictions = model.predict(x_adv.reshape(dim))[0]
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    return result != actual


  # Number of agents and decision variables
  n_agents = agents
  n_variables = dim[1] * dim[2] * dim[3]
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(0)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(0.25)

  #Creates the optimizer
  params={'model':model, 'x_clean':x_clean, 'x_adv': None,
  'y_clean': y_clean,'epsilon' : epsilon,'l_2_min':False, 'dim':dim}
  optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
  #optimizer = opytimizer.optimizers.evolutionary.GA()
  #optimizer = opytimizer.optimizers.swarm.CS()
  #optimizer = opytimizer.optimizers.swarm.PSO()
  #optimizer = opytimizer.optimizers.misc.AOA()

  space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
  function = Function(evaluate_acc)
  #function = ConstrainedFunction(evaluate_acc, [l_2_constraint], 10000.0)

  # Bundles every piece into Opytimizer class
  opt = Opytimizer(space, optimizer, function, save_agents=False)
  #Runs the optimization task
  opt.start(n_iterations = iterations)

  xopt = opt.space.best_agent.position
  x_adv = process_digit(x_clean, xopt.ravel(), epsilon, dim=dim)
  #x_adv = x_adv.reshape(dim)
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape(dim)))
  #eval_count += 1 # 1 for above prediction!
  attack_succ = np.argmax(y_clean) != adv_pred and dist <= max_l_2
  logger.info(f'Prediction Not Equal?: {np.argmax(y_clean) != adv_pred }')
  #logger.info(f"Inequality constraint count: {inequality_constraint.count}")

  # if(attack_succ == True):
  #   logger.info("Starting Phase#2 Exploitation")
  #   for i in range(1):
  #     #epsilon = epsilon
  #     logger.info(f"Restarting L2 Minimization loop: {i} with epsilon: {epsilon}")
  #     params={'model':model, 'x_clean':x_clean, 'x_adv': None,
  #     'y_clean': y_clean,'epsilon' : epsilon, 'l_2_min':True, 'dim':dim}
  #     #optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
  #     optimizer = opytimizer.optimizers.swarm.CS()
  #     #space_l_2 = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
  #     opt_l_2 = Opytimizer(space, optimizer, function, save_agents=False)
  #     #opt_l_2.space.best_agent.position = opt.space.best_agent.position
  #     #opt_l_2.space.best_agent.position = opt.space.best_agent.position
  #     #Runs the optimization task
  #     opt_l_2.start(n_iterations = round(iterations*l_2_mul))
  #     xopt_curr = opt_l_2.space.best_agent.position
  #     x_adv_curr = process_digit(x_clean, xopt_curr.ravel(), epsilon, dim=dim)
  #     x_adv_curr = x_adv_curr.reshape(dim)
  #     adv_pred_curr = np.argmax(model.predict(x_adv_curr.reshape(dim)))
  #     eval_count += 1
  #     attack_succ_curr = np.argmax(y_clean) != adv_pred_curr
  #     dist_curr = l_2_dist(x_clean, x_adv_curr)
  #     if(attack_succ_curr == True and dist_curr < dist):
  #       opt = opt_l_2
  #       x_adv = np.copy(x_adv_curr)
  #       dist = dist_curr

  all_dist = get_all_dist(x_clean, x_adv)
  logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} All Dist:{all_dist}, L2_Iters: {l2_iter}")
  return x_adv, eval_count, dist


def get_opyt_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, agents=20, l_2_mul=0.5, dim=(1,28,28,1)):
  iteration = round(iterations)
  logger.info(f"\nIterations:{iteration}, epsilon: {epsilon} and l_2_mul:{l_2_mul}")

  no_samples = len(x_test_random)
  adv_nvg = np.empty(dim)

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    logger.info(f"Generating example:{i}")
    adv_nvg[i], count, dist = get_adv_opyt_example(model,
                                                  x_test_random[i],
                                                  y_test_random[i],
                                                  epsilon = epsilon,
                                                  iterations = iterations,
                                                   max_l_2 = max_l_2,
                                                   agents = agents,
                                                   l_2_mul=l_2_mul,
                                                   dim=(1, dim[1], dim[2], dim[3])
                                                  )
    query_count.append(count)
    l_2.append(dist)

  logger.info(f'Shape of adv_nvg: {adv_nvg.shape} and shape of y_test_random:{y_test_random.shape}')
  loss, acc = model.evaluate(adv_nvg, np.argmax(y_test_random, axis=1))
  #acc = accuracy_score(y_pred_nvg, y_test_random)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  logger.info(f"Total Examples: {len(y_test_random)}, Iterations:{iterations}, espilon: {epsilon} and Max-L2:{max_l_2} Agents: {agents} l_2_mul: {l_2_mul}\nAccuracy: {acc} Mean L2 Counted: {l_2_mean} Query: {query_mean}\n")

  ##PRODUCTION
  # if (acc == 0):
  #   return -l_2_mean, l_2_mean, query_mean, adv_nvg
  # return  acc * (-l_2_mean),l_2_mean, query_mean, adv_nvg
  return  acc,l_2_mean, query_mean, adv_nvg

  # # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return np.log(l_2_mean)
  # return  np.log(acc * l_2_mean)

def get_adv_opyt_target_example(model, x_clean, y_clean,x_target, y_target,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20, l_2_mul=.5,
                        dim=(1,28,28,1)):
  eval_count = 0
  x_adv = None
  x_clean_mod = np.copy(x_clean)
  x_original = np.copy(x_clean)
  l2_iter = round(iterations*2)
  target_label = np.argmax(y_target)
  logger.info(f'Clean:{np.argmax(y_clean)} and Target:{target_label}')

  def minimize_l_2(x):
    nonlocal eval_count
    eval_count += 1
    x = x.clip(0,1)
    x = np.where(x >= 0.9, 1, x)
    x = np.where(x < 0.9, 0, x)
    x = x * 0.5
    predictions = model.predict(x.reshape(dim))[0]
    result = np.argmax(predictions)
    if(result == target_label):
      #return float(l_2_dist(x, x_original))
      logger.to_file(f'L2:{l_2_dist(x, x_original)}')
      return -float(np.count_nonzero(x == 0))
      #return l_2_dist(x, x_original)
    else:
      return float(1e10)


  def evaluate_acc(x):
    nonlocal eval_count
    nonlocal target_label
    nonlocal x_original
    eval_count += 1
    x_adv = process_image_target(x_clean_mod, x.ravel(), x_target, epsilon, dim=dim)
    predictions = model.predict(x_adv.reshape(dim))[0]
    result = np.argmax(predictions)
    if(result == target_label):
      return float(l_2_dist(x_adv, x_original))
    else:
      return 100.0

  # Number of agents and decision variables
  n_agents = agents
  n_variables = dim[1] * dim[2] * dim[3]
  #n_variables = 1
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(0)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)

  #Creates the optimizer
  params={'model':model, 'x_clean':x_clean_mod, 'x_adv': None,
  'y_clean': y_clean,'epsilon' : epsilon,'l_2_min':False, 'dim':dim}
  x_adv_l_2_xopt = None
  for i in range(5):
    logger.info(f'Starting search for initial adv image loop {i}')
    optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
    space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
    function_l_2 = Function(minimize_l_2)
    opt = Opytimizer(space, optimizer, function_l_2, save_agents=False)
    opt.start(n_iterations = round(l2_iter/8)*(i+1))
    x_adv_l_2_xopt = opt.space.best_agent.position
    x_adv_l_2_xopt = x_adv_l_2_xopt.clip(0,1)
    x_adv_l_2_xopt = np.where(x_adv_l_2_xopt >= 0.9, 1, x_adv_l_2_xopt)
    x_adv_l_2_xopt = np.where(x_adv_l_2_xopt < 0.9, 0, x_adv_l_2_xopt)
    x_adv_l_2_xopt = x_adv_l_2_xopt * 0.5
    x_adv_l_2_xopt = x_adv_l_2_xopt.reshape(dim)
    pred = np.argmax(model.predict(x_adv_l_2_xopt))
    logger.info(f'pred: {pred} and target_label:{target_label}')
    if  pred != target_label:
      logger.info(f'Couldn\'t find initial adv image. Queries:{eval_count}')
    elif l_2_dist(x_adv_l_2_xopt, x_original) > 15:
      logger.info(f'Found initial adv image with higher L2. Queries:{eval_count}')
      break
    else:
      logger.info(f'Found initial adv image within L2 limit. Queries:{eval_count}')
      break


  logger.info(f'Starting attack!')
  x_clean_mod =  x_original + x_adv_l_2_xopt * 0.05
  x_clean_mod = x_clean_mod.clip(0,1)
  eval_count +=1
  pred = np.argmax(model.predict(x_clean_mod))
  logger.info(f'After adding initial adv image, Pred: {pred},target_label:{target_label}, L2:{l_2_dist(x_original,x_clean_mod )}')
  if  pred != target_label:
    logger.info(f'Couldn\'t find adv image after adding initial adv image. Queries:{eval_count}')

  lower_bound = np.empty(n_variables)
  lower_bound.fill(-1)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)
  optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
  space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
  function = Function(evaluate_acc)
  #function = ConstrainedFunction(evaluate_acc, [l_2_constraint], 10000.0)
  opt = Opytimizer(space, optimizer, function, save_agents=False)
  #logger.info(f'Shape of x_adv_l_2_xopt:{x_adv_l_2_xopt.shape}')
  # for agent in space.agents:
  #   agent.fill_with_static(x_adv_l_2_xopt.ravel() + 0.01)
  # #Runs the optimization task
  opt.start(n_iterations = iterations)

  xopt = opt.space.best_agent.position
  x_adv = process_image_target(x_clean_mod, xopt.ravel(), x_target, epsilon, dim=dim)
  #x_adv = x_adv.reshape(dim)
  dist = l_2_dist(x_original, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape(dim)))
  #eval_count += 1 # 1 for above prediction!
  attack_succ = adv_pred==target_label and dist <= max_l_2
  logger.info(f'Prediction:{adv_pred} and Target:{target_label}')
  #logger.info(f"Inequality constraint count: {inequality_constraint.count}")
  all_dist = get_all_dist(x_original, x_adv)
  logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} All Dist:{all_dist}, L2_Iters: {l2_iter}")
  return x_adv, eval_count, dist

def get_opyt_target_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, agents=20, l_2_mul=0.5, dim=(1,28,28,1)):
  iteration = round(iterations)
  logger.info(f"\nIterations:{iteration}, epsilon: {epsilon} and l_2_mul:{l_2_mul}")

  no_samples = len(x_test_random)
  adv_nvg = np.empty(dim)
  y_all_labels = np.argmax(y_test_random, axis=1)
  logger.info(f'y_all_labels:{y_all_labels}')

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    i=1
    target = None
    logger.info(f"Generating example:{i}")
    y_clean_label = np.argmax(y_test_random[i])
    logger.info(f'y_clean_label:{y_clean_label}')
    t = np.where(y_all_labels != y_clean_label)[0]
    logger.info(f't : {t}')
    min_l_2 = 100
    min_ind = None
    for ind in t:
      curr_l_2 = l_2_dist(x_test_random[i], x_test_random[ind])
      if curr_l_2 < min_l_2:
        min_l_2 = curr_l_2
        min_ind = ind
    #target = np.random.choice(t, size=1)
    logger.info(f'min_ind:{min_ind}')
    x_target = x_test_random[min_ind]
    y_target = y_test_random[min_ind]
    adv_nvg[i], count, dist = get_adv_opyt_target_example(model,
                                                  x_test_random[i],
                                                  y_test_random[i],
                                                  x_target.reshape((dim[1], dim[2], dim[3])),
                                                  y_target,
                                                  epsilon = epsilon,
                                                  iterations = iterations,
                                                   max_l_2 = max_l_2,
                                                   agents = agents,
                                                   l_2_mul=l_2_mul,
                                                   dim=(1, dim[1], dim[2], dim[3])
                                                  )
    query_count.append(count)
    l_2.append(dist)
    break

  loss, acc = model.evaluate(adv_nvg, np.argmax(y_test_random, axis=1))
  #acc = accuracy_score(y_pred_nvg, y_test_random)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  logger.info(f"Total Examples: {len(y_test_random)}, Iterations:{iterations}, espilon: {epsilon} and Max-L2:{max_l_2} Agents: {agents} l_2_mul: {l_2_mul}\nAccuracy: {acc} Mean L2 Counted: {l_2_mean} Query: {query_mean}\n")

  ##PRODUCTION
  # if (acc == 0):
  #   return -l_2_mean, l_2_mean, query_mean, adv_nvg
  # return  acc * (-l_2_mean),l_2_mean, query_mean, adv_nvg
  return  acc,l_2_mean, query_mean, adv_nvg

  # # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return np.log(l_2_mean)
  # return  np.log(acc * l_2_mean)



def get_nvg_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, dim=(1,28,28,1)):
  iteration = round(iterations)
  logger.info(f"Iterations:{iteration}, espilon: {epsilon} and Max-L2:{max_l_2}")

  no_samples = len(x_test_random)
  adv_nvg = np.empty(dim)

  i = 0
  query_count = []
  l_2 = []
  #params = ng.p.Array(shape=(dim[1] * dim[2] * dim[3],)).set_bounds(0, 0.25)

  for i in range(no_samples):
    logger.info(f'Generating example:{i}')
    params = ng.p.Array(shape=(dim[1] * dim[2] * dim[3],)).set_bounds(0, 0.05)
    #params.value = np.zeros_like(x_test_random[i].ravel())

    optimizer = ng.optimizers.AlmostRotationInvariantDE(budget=iterations,
                                 parametrization=params)
    # DEthenPSO = ng.optimizers.Chaining([
    #                                      ng.optimizers.AlmostRotationInvariantDE,
    #                                      ng.optimizers.RealSpacePSO], [iterations]
    #                                     )
    # optimizer = DEthenPSO(budget=iterations, parametrization=params)

    optimizer.parametrization.register_cheap_constraint(
       lambda x: l_2_dist(
           process_digit(x_test_random[i], x, epsilon, dim=(1, dim[1], dim[2], dim[3])), x_test_random[i]) < max_l_2)

    adv_nvg[i], count, dist = get_adv_nvg_example(model, optimizer,
                                                  x_test_random[i],
                                                  y_test_random[i] ,
                                                  epsilon = epsilon,
                                                  max_l_2 = max_l_2,
                                                  dim=(1, dim[1], dim[2], dim[3])
                                                  )
    query_count.append(count)
    l_2.append(dist)

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = get_accuracy(y_pred_nvg, y_test_random)
  #l_2 = get_dataset_l_2_dist(x_test_random, x_test_nvg)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  logger.info(f"\nTotal Examples: {len(y_test_random)}, Iterations:{iterations}, espilon: {epsilon} and Max-L2:{max_l_2} \nAccuracy: {acc} Mean L2 Counted: {l_2_mean} Query: {query_mean}")

  #PRODUCTION
  if (acc == 0):
    return -l_2_mean, l_2_mean, query_mean, x_test_nvg
  return  acc * (-l_2_mean),l_2_mean, query_mean, x_test_nvg

  # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return np.log(l_2_mean)
  # return  np.log(acc * l_2_mean)

def get_adv_nvg_example(model,optimizer, x_clean_mod, y_clean,
                        epsilon = 0.5, max_l_2=3, dim=(1,28,28,1)):
  # @counter
  # def evaluate_acc(x):
  #   x_adv = process_digit(x_clean, x, epsilon)
  #   predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
  #   result = np.argmax(predictions)
  #   actual = np.argmax(y_clean)
  #   dist = float(l_2_dist(x_clean, x_adv))
  #   if(result != actual):
  #     return float(dist)
  #     logger.to_file(f'result:{result}, dist:{dist}')
  #   else:
  #     fitness = float(predictions[actual] * 100 * dist)
  #     logger.to_file(f'result:{result}, dist:{dist} fitness:{fitness}')
  #     return float(predictions[actual] * 100 * dist)
  #   #return np.log(predictions[np.argmax(y_clean)]) # * l_2_dist(x_clean, x_adv))
  #   #return np.log(predictions[np.argmax(y_clean)]) #* l_2_dist(x_clean, x_adv))

  @counter
  def evaluate_acc(x):
    x_adv = process_digit(x_clean, x.ravel(), epsilon, dim=dim)
    predictions = model.predict(x_adv.reshape(dim))[0]
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    dist = float(l_2_dist(x_clean, x_adv))
    #dist = np.exp(-(l_2_dist(x_clean, x_adv)**2)/2)
    if(result != actual):
      return float(dist)
    else:
      #return float(10*np.amax(predictions) + 10)
      return 100

  xopt = optimizer.minimize(evaluate_acc)
  x = np.array(xopt.value)
  x_adv = process_digit(x_clean, x, epsilon, dim=dim)
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape(dim)))
  attack_succ = np.argmax(y_clean) != adv_pred
  all_dist = get_all_dist(x_clean, x_adv)
  logger.info(f"Attack result:{attack_succ}, Queries: {evaluate_acc.count} Dist:{dist} All dist: {all_dist}")
  return x_adv, evaluate_acc.count, dist


def get_adv_scipy_example(model,optimizer, x_clean, y_clean,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20):
  eval_count = 0

  def evaluate_acc(x):
    nonlocal eval_count
    eval_count += 1
    x_adv = process_digit(x_clean, x.ravel(), epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    if(result != actual):
      #print("SUCCESS:Actual:{} Predicted:{}".format(actual, result))
      return float(predictions[actual] * (-100) * l_2_dist(x_clean, x_adv))
      #return -1
    else:
      #print("NO SUCCESS:Actual:{} Predicted:{}".format(actual, result))
      return float(predictions[actual] * l_2_dist(x_clean, x_adv))
      #return 1
    #return float(result) # * l_2_dist(x_clean, x_adv))
    #return np.log(predictions[np.argmax(y_clean)]) #* l_2_dist(x_clean, x_adv))

  # define range for input
  r_min, r_max = 0, 1
  # define the starting point as a random sample from the domain
  pt = r_min + np.random.rand(784) * (r_max - r_min)
  xopt = optimizer(evaluate_acc, pt, stepsize=0.5, niter=iterations)
  x_adv = process_digit(x_clean, xopt['x'].ravel(), epsilon)
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
  attack_succ = np.argmax(y_clean) != adv_pred
  print("Attack result:{}, Queries: {} Dist:{}".format(attack_succ,
                                                       eval_count, dist))
  return x_adv, eval_count, dist

def get_scipy_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, agents=20):
  import scipy
  iteration = round(iterations)
  print("Iterations:{}, espilon: {} and Max-L2:{}".format(
      iteration, epsilon, max_l_2))

  no_samples = len(x_test_random)
  adv_nvg = np.empty((no_samples,28,28,1))

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    print("Generating example: ", i)
    # Creates the optimizer

    optimizer = scipy.optimize.basinhopping
    adv_nvg[i], count, dist = get_adv_scipy_example(model, optimizer,
                                                  x_test_random[i],
                                                  y_test_random[i],
                                                  epsilon = epsilon,
                                                  iterations = iterations,
                                                   max_l_2 = max_l_2,
                                                   agents = agents
                                                  )
    query_count.append(count)
    l_2.append(dist)

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = get_accuracy(y_pred_nvg, y_test_random)
  #l_2 = get_dataset_l_2_dist(x_test_random, x_test_nvg)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  print("\nTotal Examples: {}, Iterations:{}, espilon: {} and Max-L2:{} Agents: {}\nAccuracy: {} Mean L2 Counted: {} Query: {}".format(
      len(y_test_random), iterations, epsilon, max_l_2,agents, acc, l_2_mean,query_mean,
      l_2_dist(x_test_random.ravel(), x_test_nvg.ravel())))

  ##PRODUCTION
  # if (acc == 0):
  #   return -l_2_mean, l_2_mean, query_mean, x_test_nvg
  # return  acc * (-l_2_mean),l_2_mean, query_mean, x_test_nvg
  return  acc,l_2_mean, query_mean, x_test_nvg

  # # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return np.log(l_2_mean)
  # return  np.log(acc * l_2_mean)

def get_adv_agfree_example(model, x_clean, y_clean,
                        epsilon = 0.5, iterations=100, max_l_2=6, step=0.001):
  eval_count = 0

  def evaluate_acc(x):
    nonlocal eval_count
    eval_count += 1
    x_adv = process_digit(x_clean, x, epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    if(result != actual):
      #print("SUCCESS:Actual:{} Predicted:{}".format(actual, result))
      return -float(predictions[actual] * (-100) * l_2_dist(x_clean, x_adv))
      #return -1
    else:
      #print("NO SUCCESS:Actual:{} Predicted:{}".format(actual, result))
      return float(predictions[actual] * l_2_dist(x_clean, x_adv))
      #return 1
    #return float(result) # * l_2_dist(x_clean, x_adv))
    #return np.log(predictions[np.argmax(y_clean)]) #* l_2_dist(x_clean, x_adv))
  n_iter = 0
  x_adv = None
  attack_succ = None
  dist = None
  while (n_iter < iterations):
    xprop = np.random.randn(784) * step
    print("xprop shape ", xprop.shape)
    print("n_iter: ", n_iter)
    x_adv = np.copy(x_clean.ravel())
    x_adv = x_adv + xprop * epsilon
    dist = l_2_dist(x_clean, x_adv)
    adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
    attack_succ = np.argmax(y_clean) != adv_pred
    if (attack_succ == True):
      break
    n_iter += 1
  print("Attack result:{}, Queries: {} Dist:{}".format(attack_succ,
                                                       n_iter, dist))
  return x_adv.reshape(28,28,1), eval_count, dist

def get_agfree_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, step=0.001):
  iteration = round(iterations)
  print("Iterations:{}, espilon: {} and Max-L2:{}".format(
      iteration, epsilon, max_l_2))

  no_samples = len(x_test_random)
  adv_nvg = np.empty((no_samples,28,28,1))

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    print("Generating example: ", i)
    # Creates the optimizer
    adv_nvg[i], count, dist = get_adv_agfree_example(model,
                                                  x_test_random[i],
                                                  y_test_random[i],
                                                  epsilon = epsilon,
                                                  iterations = iterations,
                                                  max_l_2 = max_l_2,
                                                  step = 0.001
                                                  )
    query_count.append(count)
    l_2.append(dist)

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = get_accuracy(y_pred_nvg, y_test_random)
  #l_2 = get_dataset_l_2_dist(x_test_random, x_test_nvg)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  print("\nTotal Examples: {}, Iterations:{}, espilon: {} and Max-L2:{} step: {}\nAccuracy: {} Mean L2 Counted: {} Query: {}".format(
      len(y_test_random), iterations, epsilon, max_l_2,step, acc, l_2_mean,query_mean,
      l_2_dist(x_test_random.ravel(), x_test_nvg.ravel())))

  ##PRODUCTION
  # if (acc == 0):
  #   return -l_2_mean, l_2_mean, query_mean, x_test_nvg
  # return  acc * (-l_2_mean),l_2_mean, query_mean, x_test_nvg
  return  acc,l_2_mean, query_mean, x_test_nvg

  # # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return np.log(l_2_mean)
  # return  np.log(acc * l_2_mean)



def get_adv_tlbo_example(model, x_clean, y_clean,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20, l_2_step=1.5):
  eval_count = 0
  x_adv = None

  @counter
  def evaluate_acc(x):
    nonlocal eval_count
    eval_count += 1
    x_adv = process_digit(x_clean, x.ravel(), epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    dist = float(l_2_dist(x_clean, x_adv))
    #dist = float(wasserstein_distance(x_clean.ravel(), x_adv.ravel()))
    if(result != actual):
      return float(dist)
    else:
      return float(predictions[actual] * 1000)

  # Number of agents and decision variables
  n_agents = agents
  n_variables = 784
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(0)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)

  #Creates the optimizer
  tlbo = TLBO(evaluate_acc, lower_bound, upper_bound, n_population=n_agents)
  tlbo = helper_n_generations(tlbo, iterations)
  xopt, best_fitness = tlbo.best()
  x_adv = process_digit(x_clean, xopt.ravel(), epsilon)
  x_adv = x_adv.reshape((28,28,1))
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
  eval_count += iterations + 1 # 1 for above prediction!
  attack_succ = np.argmax(y_clean) != adv_pred
  logger.info(f"\nResult: Attack result:{attack_succ},Fitness: {best_fitness}, Queries: {eval_count}, Dist:{dist}\n")

  #all_dist = get_all_dist(x_clean, x_adv)
  #logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} All Dist:{all_dist}")
  return x_adv, eval_count, dist


def get_tlbo_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, agents=20, l_2_step=0.1):
  iteration = round(iterations)
  logger.info(f"\nIterations:{iteration}, epsilon: {epsilon} and l_2_step:{l_2_step}")

  no_samples = len(x_test_random)
  adv_nvg = np.empty((no_samples,28,28,1))

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    logger.info(f"\nGenerating example:{i}")
    adv_nvg[i], count, dist = get_adv_tlbo_example(model,
                                                  x_test_random[i],
                                                  y_test_random[i],
                                                  epsilon = epsilon,
                                                  iterations = iterations,
                                                   max_l_2 = max_l_2,
                                                   agents = agents,
                                                   l_2_step=l_2_step
                                                  )
    query_count.append(count)
    l_2.append(dist)

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = get_accuracy(y_pred_nvg, y_test_random)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  logger.info(f"\nTotal Examples: {len(y_test_random)}, Iterations:{iterations}, espilon: {epsilon} and Max-L2:{max_l_2} Agents: {agents} l_2_step: {l_2_step}\nAccuracy: {acc} Mean L2 Counted: {l_2_mean} Query: {query_mean}")

  ##PRODUCTION
  # if (acc == 0):
  #   return -l_2_mean, l_2_mean, query_mean, x_test_nvg
  # return  acc * (-l_2_mean),l_2_mean, query_mean, x_test_nvg
  return  acc,l_2_mean, query_mean, x_test_nvg

  # # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return np.log(l_2_mean)
  # return  np.log(acc * l_2_mean)


def get_adv_opyt_cso_example(model, x_clean, y_clean,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20, l_2_step=1.5):
  eval_count = 0
  x_adv = None

  def evaluate_acc(x):
    nonlocal eval_count
    eval_count += 1
    x_adv = process_digit(x_clean, x.ravel(), epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    dist = float(l_2_dist(x_clean, x_adv))
    #dist = float(wasserstein_distance(x_clean.ravel(), x_adv.ravel()))
    if(result != actual):
      return float(dist)
    else:
      return float(predictions[actual] * 100)

  # Number of agents and decision variables
  n_agents = agents
  n_variables = 784
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(0)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)

  #Creates the optimizer
  optimizer = opytimizer.optimizers.swarm.CS()
  #optimizer = opytimizer.optimizers.misc.AOA()

  space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
  function = Function(evaluate_acc)
  #function = ConstrainedFunction(evaluate_acc, [l_2_constraint], 10000.0)

  # Bundles every piece into Opytimizer class
  opt = Opytimizer(space, optimizer, function, save_agents=False)
  #Runs the optimization task
  opt.start(n_iterations = iterations)

  xopt = opt.space.best_agent.position
  x_adv = process_digit(x_clean, xopt.ravel(), epsilon)
  x_adv = x_adv.reshape((28,28,1))
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
  eval_count += iterations + 1 # 1 for above prediction!
  attack_succ = np.argmax(y_clean) != adv_pred
  logger.info(f"\nResult: Attack result:{attack_succ}, Queries: {eval_count} Dist:{dist}\n")

  #all_dist = get_all_dist(x_clean, x_adv)
  #logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} All Dist:{all_dist}")
  return x_adv, eval_count, dist


def get_opyt_cso_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, agents=20, l_2_step=0.1):
  iteration = round(iterations)
  logger.info(f"\nIterations:{iteration}, epsilon: {epsilon} and l_2_step:{l_2_step}")
  logger.to_file(f"\nIterations:{iteration}, epsilon: {epsilon} and l_2_step:{l_2_step}")

  no_samples = len(x_test_random)
  adv_nvg = np.empty((no_samples,28,28,1))

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    logger.info(f"\nGenerating example:{i}")
    adv_nvg[i], count, dist = get_adv_opyt_cso_example(model,
                                                  x_test_random[i],
                                                  y_test_random[i],
                                                  epsilon = epsilon,
                                                  iterations = iterations,
                                                   max_l_2 = max_l_2,
                                                   agents = agents,
                                                   l_2_step=l_2_step
                                                  )
    query_count.append(count)
    l_2.append(dist)

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = get_accuracy(y_pred_nvg, y_test_random)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  logger.info(f"\nTotal Examples: {len(y_test_random)}, Iterations:{iterations}, espilon: {epsilon} and Max-L2:{max_l_2} Agents: {agents} l_2_step: {l_2_step}\nAccuracy: {acc} Mean L2 Counted: {l_2_mean} Query: {query_mean}")

  ##PRODUCTION
  # if (acc == 0):
  #   return -l_2_mean, l_2_mean, query_mean, x_test_nvg
  # return  acc * (-l_2_mean),l_2_mean, query_mean, x_test_nvg
  return  acc,l_2_mean, query_mean, x_test_nvg

  # # #MAXIMIZE
  # if (acc == 0):
  #   return -l_2_mean
  # return  acc * (-l_2_mean)
  # # ##MINIMIZE
  # if (acc == 0):
  #   return np.log(l_2_mean)
  # return  np.log(acc * l_2_mean)

def get_adv_opyt_imagenet_example(model, x_clean, y_clean,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20, l_2_mul=.5,
                        dim=(1,28,28,1)):
  eval_count = 0
  x_adv = None
  l2_iter = round(iterations)

  def evaluate_acc(x):
    nonlocal eval_count
    eval_count += 1
    x_adv = process_imagenet(x_clean, x.ravel(), epsilon, dim=dim)
    result = get_imagenet_top_1_pred(model, x_adv)
    dist = float(l_2_dist(x_clean, x_adv))
    #dist = np.exp(-(l_2_dist(x_clean, x_adv)**2)/2)
    if(result != y_clean):
      # if (dist > max_l_2):
      #   return float(dist) * 10
      # else:
      #   return float(dist)
      return float(dist)
    else:
      #predictions.sort()
      #return float(10*(predictions[-1] - predictions[-2]) + 10 * dist)
      #return float(10*np.amax(predictions) + 10)
      return 100


  # Number of agents and decision variables
  n_agents = agents
  n_variables = dim[1] * dim[2] * dim[3]
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(-255)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(255)

  #Creates the optimizer
  params={'model':model, 'x_clean':x_clean, 'x_adv': None,
  'y_clean': y_clean,'epsilon' : epsilon,'l_2_min':False, 'dim':dim}
  optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
  #optimizer = opytimizer.optimizers.evolutionary.GA()
  #optimizer = opytimizer.optimizers.swarm.CS()
  #optimizer = opytimizer.optimizers.swarm.PSO()
  #optimizer = opytimizer.optimizers.misc.AOA()

  space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
  function = Function(evaluate_acc)
  #function = ConstrainedFunction(evaluate_acc, [l_2_constraint], 10000.0)

  # Bundles every piece into Opytimizer class
  opt = Opytimizer(space, optimizer, function, save_agents=False)
  #Runs the optimization task
  opt.start(n_iterations = iterations)

  xopt = opt.space.best_agent.position
  x_adv = process_imagenet(x_clean, xopt.ravel(), epsilon, dim=dim)
  #x_adv = x_adv.reshape(dim)
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = get_imagenet_top_1_pred(model, x_adv)
  #eval_count += 1 # 1 for above prediction!
  attack_succ = y_clean != adv_pred and dist <= max_l_2
  logger.info(f'Prediction Not Equal?: {y_clean != adv_pred } and dist:{dist}')
  logger.info(f'y_clean:{y_clean} adv_pred:{adv_pred}')

  all_dist = get_all_dist(x_clean, x_adv)
  logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} All Dist:{all_dist}, L2_Iters: {l2_iter}")
  return x_adv, eval_count, dist

def get_adv_opyt_target_imagenet_example(model, x_clean, y_clean,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20, l_2_mul=.5,
                        dim=(1,28,28,1)):
  eval_count = 0
  x_adv = None
  x_clean_mod = np.copy(x_clean)
  x_original = np.copy(x_clean)
  l2_iter = round(iterations*2)

  def minimize_l_2(x):
    nonlocal eval_count
    eval_count += 1
    #x = x.clip(0,255)
    # x = np.where(x >= 200, 100, x)
    # x = np.where(x < 200, 0, x)
    #x = x * 0.01
    x_adv = process_image_target_imagenet(x_original, x.ravel(), epsilon, dim=dim)
    result = get_imagenet_top_1_pred(model, x_adv.reshape((dim)))
    dist = l_2_dist(x_adv, x_original)
    if(result != y_clean):
      #return float(l_2_dist(x, x_original))
      logger.to_file(f'L2:{dist}')
      #return -float(np.count_nonzero(x == 0))
      return l_2_dist(x_adv, x_original)
    else:
      return float(1e10)


  def evaluate_acc(x):
    nonlocal eval_count
    nonlocal x_original
    eval_count += 1
    x_adv = process_imagenet(x_clean_mod, x.ravel(), epsilon, dim=dim)
    result = get_imagenet_top_1_pred(model, x_adv)
    dist = l_2_dist(x_adv, x_original)
    logger.to_file(f'L2:{dist}')
    if(result != y_clean):
      return float(dist)
    else:
      return float(1e10)

  # Number of agents and decision variables
  n_agents = agents
  n_variables = dim[1] * dim[2] * dim[3]
  #n_variables = 1
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(-0.5)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)

  #Creates the optimizer
  params={'model':model, 'x_clean':x_clean_mod, 'x_adv': None,
  'y_clean': y_clean,'epsilon' : epsilon,'l_2_min':False, 'dim':dim}
  x_adv_l_2_xopt = None
  best_x_adv_l_2_xopt = None
  best_dist = 60000
  max_l_2_dist = 30000
  for i in range(5):
    logger.info(f'Starting search for initial adv image loop {i} with best dist:{best_dist}')
    optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
    space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
    function_l_2 = Function(minimize_l_2)
    opt = Opytimizer(space, optimizer, function_l_2, save_agents=False)
    #opt.start(n_iterations = round(l2_iter/8)*(i+1))
    opt.start(n_iterations = round(iterations))
    x_adv_l_2_xopt = opt.space.best_agent.position
    #x_adv_l_2_xopt = x_adv_l_2_xopt.clip(0,255)
    # x_adv_l_2_xopt = np.where(x_adv_l_2_xopt >= 200, 100, x_adv_l_2_xopt)
    # x_adv_l_2_xopt = np.where(x_adv_l_2_xopt < 200, 0, x_adv_l_2_xopt)
    # #x_adv_l_2_xopt = x_adv_l_2_xopt * 0.01
    x_adv_l_2_xopt = x_adv_l_2_xopt.reshape(dim)
    x_adv_l_2_xopt = process_image_target_imagenet(x_original, x_adv_l_2_xopt.ravel(), epsilon, dim=dim)
    pred = get_imagenet_top_1_pred(model, x_adv_l_2_xopt)
    dist = l_2_dist(x_adv_l_2_xopt, x_original)
    logger.info(f'pred: {pred} and dist:{dist}')
    if (best_x_adv_l_2_xopt is None):
      best_x_adv_l_2_xopt = x_adv_l_2_xopt

    if pred == y_clean:
      logger.info(f'Couldn\'t find initial adv image. L2:{dist} Queries:{eval_count}')
    elif dist < best_dist:
      best_dist = dist
      best_x_adv_l_2_xopt = x_adv_l_2_xopt
      logger.info(f'Found initial adv image with higher, best L2:{best_dist}. Queries:{eval_count}')
      break
      if dist < max_l_2_dist:
        logger.info(f'Breaking from initial search with L2:{best_dist}')
        break

  return best_x_adv_l_2_xopt, eval_count, dist

  logger.info(f'Starting attack!')
  logger.info(f'Selecting initial adv image with L2:{best_dist}')
  x_clean_mod =  x_original + best_x_adv_l_2_xopt * 0.1
  x_clean_mod = x_clean_mod.clip(0,255)
  eval_count += 1
  pred = get_imagenet_top_1_pred(model, x_clean_mod)
  logger.info(f'After adding initial adv image, Pred: {pred}, L2:{l_2_dist(x_original,x_clean_mod )}')
  if  pred == y_clean:
    logger.info(f'Couldn\'t find adv image after adding initial adv image. Queries:{eval_count}')

  lower_bound = np.empty(n_variables)
  lower_bound.fill(-1)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)
  optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
  space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
  function = Function(evaluate_acc)
  #function = ConstrainedFunction(evaluate_acc, [l_2_constraint], 10000.0)
  opt = Opytimizer(space, optimizer, function, save_agents=False)
  #logger.info(f'Shape of x_adv_l_2_xopt:{x_adv_l_2_xopt.shape}')
  # for agent in space.agents:
  #   agent.fill_with_static(x_adv_l_2_xopt.ravel() + 0.01)
  # #Runs the optimization task
  opt.start(n_iterations = iterations)

  xopt = opt.space.best_agent.position
  x_adv = process_imagenet(x_clean_mod, xopt.ravel(), epsilon, dim=dim)
  #x_adv = x_adv.reshape(dim)
  dist = l_2_dist(x_original, x_adv)
  adv_pred = get_imagenet_top_1_pred(model, x_adv)
  #eval_count += 1 # 1 for above prediction!
  attack_succ = adv_pred != y_clean and dist <= max_l_2
  logger.info(f'Predictions not euqal?: {adv_pred != y_clean} Adv Prediction:{adv_pred}')
  #logger.info(f"Inequality constraint count: {inequality_constraint.count}")
  all_dist = get_all_dist(x_original, x_adv)
  logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} All Dist:{all_dist}, L2_Iters: {l2_iter}")
  return x_adv, eval_count, dist
