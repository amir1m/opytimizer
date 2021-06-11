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
                         n=10, epsilon=0.001, seed=SEED):
  #np.random.seed(SEED)
  x_adv = {}
  classifier = KerasClassifier(model) #For ART attacks

  x_test_random, y_test_random, rand_indices = get_random_correct_samples(
      n, x_test, y_test, model.predict(x_test))
  x_adv['CLEAN_X'] = x_test_random
  x_adv['CLEAN_Y'] = y_test_random

  for attack in attack_list:
    if(attack == 'FGSM'):
      print("\nGenerating adv examples using attack FGSM")
      epsilon = 0.1  # Maximum perturbation
      adv_crafter = FastGradientMethod(classifier, eps=epsilon)
      x_adv[attack] = adv_crafter.generate(x=x_test_random)

    if(attack == 'CWL2'):
      print("\nGenerating adv examples using attack CWL2")
      #epsilon = 0.1  # Maximum perturbation
      adv_crafter = CarliniL2Method(classifier)
      x_adv[attack] = adv_crafter.generate(x=x_test_random)

    elif(attack == 'BOUNDARY'):
      print("\nGenerating adv examples using attack BOUNDARY")
      boundary = BoundaryAttack(classifier, targeted=False)
      x_adv[attack] = boundary.generate(x_test_random)

    elif(attack == 'ZOO'):
      print("\nGenerating adv examples using attack ZOO")
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
      x_adv[attack] = zoo.generate(x_test_random)

    elif(attack == 'SIMBA'):
      print("\nGenerating adv examples using attack SIMBA")
      simba = SimBA(classifier)
      x_adv[attack] = simba.generate(x_test_random)

    elif(attack == 'HOPSKIPJUMP'):
      print("\nGenerating adv examples using attack HOPSKIPJUMP")
      hopskipjump = HopSkipJump(classifier)
      x_adv[attack] = hopskipjump.generate(x_test_random)

    elif(attack == 'CSO'):
      print("\nGenerating adv examples using attack CSO")
      #Already tuned hyper-parameters
      loss, l_2_mean, query_mean, x_test_cso = get_cso_adv(model, x_test_random,
                                                     y_test_random, n=800, pa=.25,
                                                    iterations=1, epsilon=epsilon)
      x_adv[attack] = x_test_cso
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
  x_clean_ravel = (x_clean_ravel-min(x_clean_ravel)) / (max(x_clean_ravel)-min(x_clean_ravel))
  return x_clean_ravel.reshape(dim)

# TAKES ARGUMENTS
def get_cso_adv(model, x_test_random, y_test_random,
                n=150, iterations = 1, pa=0.5, nest=784, epsilon = 3.55, max_l_2=4):
  iteration = round(iterations)
  n = round(n)
  print("n: {} Iteration:{} and espilon: {}".format(n,iteration, epsilon))

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
      if (dist > max_l_2):
        return float(dist) * 10
      else:
        return float(dist)
    else:
      predictions.sort()
      return float(10*(predictions[-1] - predictions[-2]) + 10 * dist)

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
  upper_bound.fill(1)

  #Creates the optimizer
  params={'model':model, 'x_clean':x_clean, 'x_adv': None,
  'y_clean': y_clean,'epsilon' : epsilon,'l_2_min':False, 'dim':dim}
  optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
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
  eval_count += 1 # 1 for above prediction!
  attack_succ = np.argmax(y_clean) != adv_pred
  logger.info(f"Exploration Phase#1 Result: Attack result:{attack_succ}, Queries: {eval_count} Dist:{dist}\n")
  #logger.info(f"Inequality constraint count: {inequality_constraint.count}")

  if(attack_succ == True):
    logger.info("Starting Phase#2 Exploitation")
    for i in range(1):
      #epsilon = epsilon
      logger.info(f"Restarting L2 Minimization loop: {i} with epsilon: {epsilon}")
      params={'model':model, 'x_clean':x_clean, 'x_adv': None,
      'y_clean': y_clean,'epsilon' : epsilon, 'l_2_min':True, 'dim':dim}
      #optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
      optimizer = opytimizer.optimizers.swarm.CS()
      #space_l_2 = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
      opt_l_2 = Opytimizer(space, optimizer, function, save_agents=False)
      #opt_l_2.space.best_agent.position = opt.space.best_agent.position
      #opt_l_2.space.best_agent.position = opt.space.best_agent.position
      #Runs the optimization task
      opt_l_2.start(n_iterations = round(iterations*l_2_mul))
      xopt_curr = opt_l_2.space.best_agent.position
      x_adv_curr = process_digit(x_clean, xopt_curr.ravel(), epsilon, dim=dim)
      x_adv_curr = x_adv_curr.reshape(dim)
      adv_pred_curr = np.argmax(model.predict(x_adv_curr.reshape(dim)))
      eval_count += 1
      attack_succ_curr = np.argmax(y_clean) != adv_pred_curr
      dist_curr = l_2_dist(x_clean, x_adv_curr)
      if(attack_succ_curr == True and dist_curr < dist):
        opt = opt_l_2
        x_adv = np.copy(x_adv_curr)
        dist = dist_curr

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
    logger.info(f"\n\nGenerating example:{i}")
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

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = get_accuracy(y_pred_nvg, y_test_random)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  logger.info(f"\nTotal Examples: {len(y_test_random)}, Iterations:{iterations}, espilon: {epsilon} and Max-L2:{max_l_2} Agents: {agents} l_2_mul: {l_2_mul}\nAccuracy: {acc} Mean L2 Counted: {l_2_mean} Query: {query_mean}")

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

def get_adv_opyt_target_example(model, x_clean, y_clean, target=0,
                        epsilon = 0.5, iterations=100, max_l_2=6, agents=20, l_2_mul=.5):
  eval_count = 0
  x_adv = None
  l2_iter = round(iterations)
  logger.info(f'Clean: {y_clean}, Generating Target {target}')

  def evaluate_acc(x):
    nonlocal eval_count
    eval_count += 1
    x_adv = process_digit(x_clean, x.ravel(), epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    target_pred = predictions[target]
    #logger.info(f'Predictions: {predictions}, Target Predictions: {target_pred}')
    result = np.argmax(predictions)
    actual = np.argmax(y_clean)
    # #logger.to_file(f'Result: {result}, target: {target}')
    dist = float(l_2_dist(x_clean, x_adv))
    if(result == target):
      return float(dist)
    else:
      return 100 - float(target_pred)
    # return -float(target_pred)

  # Number of agents and decision variables
  n_agents = agents
  n_variables = 784
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(0)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)

  #Creates the optimizer
  params={'model':model, 'x_clean':x_clean, 'x_adv': None,
  'y_clean': y_clean,
  'epsilon' : epsilon,'l_2_min':False}
  optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
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
  adv_pred = model.predict(x_adv.reshape((1,28,28,1)))[0]
  eval_count += 1
  adv_pred_label = np.argmax(adv_pred)
  before_attack_pred = model.predict(x_clean.reshape((1,28,28,1)))[0]
  before_attack_label = np.argmax(before_attack_pred)
  logger.info(f'Before Attack: Clean Prediction: {before_attack_pred[np.argmax(y_clean)]} Target Prediction: {before_attack_pred[target]} Label: {before_attack_label}')
  logger.info(f'After Attack: Clean Prediction: {adv_pred[np.argmax(y_clean)]} Target Prediction: {adv_pred[target]} Label:{adv_pred_label}')
  eval_count += 1 # 1 for above prediction!
  attack_succ =  adv_pred_label == target
  #logger.info(f"\nResult: Target Attack result:{attack_succ}, Queries: {eval_count} Dist:{dist}\n")

  if(attack_succ == True):
    logger.info("\nStarting Phase#2 Exploitation\n")
    for i in range(1):
      logger.info(f"\nRestarting L2 Minimization loop: {i}")
      params={'model':model, 'x_clean':x_clean, 'x_adv': None,
      'y_clean': y_clean,'epsilon' : epsilon, 'l_2_min':True}
      #optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
      optimizer = opytimizer.optimizers.swarm.CS()
      #space_l_2 = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
      opt_l_2 = Opytimizer(space, optimizer, function, save_agents=False)
      #opt_l_2.space.best_agent.position = opt.space.best_agent.position
      #opt_l_2.space.best_agent.position = opt.space.best_agent.position
      #Runs the optimization task
      opt_l_2.start(n_iterations = round(iterations))
      xopt_curr = opt_l_2.space.best_agent.position
      x_adv_curr = process_digit(x_clean, xopt_curr.ravel(), epsilon)
      x_adv_curr = x_adv_curr.reshape((28,28,1))
      adv_pred_curr = np.argmax(model.predict(x_adv_curr.reshape((1,28,28,1)))[0])
      eval_count += 1
      attack_succ_curr = target == adv_pred_curr
      dist_curr = l_2_dist(x_clean, x_adv_curr)
      if(attack_succ_curr == True and dist_curr < dist):
        opt = opt_l_2
        x_adv = np.copy(x_adv_curr)
        dist = dist_curr

  # if(attack_succ == True):
  #   logger.info('Starting L2 Minimization\n')
  #   xopt_curr = minimize(evaluate_acc, x_adv.ravel(), method='nelder-mead',
  #              options={'xatol': 1e-8, 'disp': True})
  #   x_adv_curr = process_digit(x_clean, xopt_curr.ravel(), epsilon)
  #   x_adv_curr = x_adv_curr.reshape((28,28,1))
  #   adv_pred_curr = np.argmax(model.predict(x_adv_curr.reshape((1,28,28,1))))
  #   eval_count += l2_iter + 1
  #   attack_succ_curr = np.argmax(y_clean) != adv_pred_curr
  #   dist_curr = l_2_dist(x_clean, x_adv_curr)
  #   if(attack_succ_curr == True and dist_curr < dist):
  #     opt = opt_l_2
  #     x_adv = np.copy(x_adv_curr)
  #     dist = dist_curr

  # dist = l_2_dist(x_clean, x_adv)
  # adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
  # attack_succ = np.argmax(y_clean) != adv_pred

  all_dist = get_all_dist(x_clean, x_adv)
  logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} All Dist:{all_dist}, L2_Iters: {l2_iter}")
  return x_adv, eval_count, dist


def get_opyt_target_adv(model, x_test_random, y_test_random, y_target,
                iterations = 100, epsilon = 3.55, max_l_2=6, agents=20, l_2_mul=0.5):
  iteration = round(iterations)
  logger.info(f"\nIterations:{iteration}, epsilon: {epsilon} and l_2_mul:{l_2_mul}")

  no_samples = len(x_test_random)
  adv_nvg = np.empty((no_samples,28,28,1))

  i = 0
  query_count = []
  l_2 = []
  for i in range(no_samples):
    logger.info(f"\nGenerating example:{i}")
    adv_nvg[i], count, dist = get_adv_opyt_target_example(model,
                                                  x_test_random[i],
                                                  y_test_random[i],
                                                  y_target[i],
                                                  epsilon = epsilon,
                                                  iterations = iterations,
                                                   max_l_2 = max_l_2,
                                                   agents = agents,
                                                   l_2_mul=l_2_mul
                                                  )
    query_count.append(count)
    l_2.append(dist)

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = (np.sum(np.argmax(y_pred_nvg, axis=1) == y_target) / len(y_target)) * 100
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  logger.info(f"\nTotal Examples: {len(y_test_random)}, Iterations:{iterations}, espilon: {epsilon} and Max-L2:{max_l_2} Agents: {agents} l_2_mul: {l_2_mul}\nASR: {acc} Mean L2 Counted: {l_2_mean} Query: {query_mean}")

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



def get_nvg_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6, dim=(1,28,28,1)):
  iteration = round(iterations)
  logger.info(f"Iterations:{iteration}, espilon: {epsilon} and Max-L2:{max_l_2}")

  no_samples = len(x_test_random)
  adv_nvg = np.empty(dim)

  i = 0
  query_count = []
  l_2 = []
  params = ng.p.Array(shape=(dim[1] * dim[2] * dim[3],)).set_bounds(0, 1)

  for i in range(no_samples):
    logger.info(f'Generating example:{i}')
    params = ng.p.Array(shape=(dim[1] * dim[2] * dim[3],)).set_bounds(0, 1)
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

def get_adv_nvg_example(model,optimizer, x_clean, y_clean,
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
      if (dist > max_l_2):
        return float(dist) * 10
      else:
        return float(dist)
    else:
      predictions.sort()
      return float(10*(predictions[-1] - predictions[-2]) + 10 * dist)

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
