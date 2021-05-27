"""# Attack Function

## Driver Functions
"""
SEED =42
import opytimizer.utils.logging as l
logger = l.get_logger(__name__)

import numpy as np


from attack_utils import *


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

def process_digit(x_clean, x_prop, epsilon):
  x_clean_ravel = np.copy(x_clean.ravel())
  x_clean_ravel += x_prop * epsilon
  x_clean_ravel = (x_clean_ravel-min(x_clean_ravel)) / (max(x_clean_ravel)-min(x_clean_ravel))
  return x_clean_ravel.reshape((28,28,1))

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

def get_adv_opyt_example(model,optimizer, x_clean, y_clean,
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
    if(result != actual):
      #print("SUCCESS:Actual:{} Predicted:{}".format(actual, result))
      #return float(predictions[actual] * (-100) * l_2_dist(x_clean, x_adv))
      #return float(predictions[actual]  * l_2_dist(x_clean, x_adv))
      fitness = float(l_2_dist(x_clean, x_adv))
      logger.to_file(f'fitness in evaluate_acc {fitness}')
      return fitness
      #return -1
    else:
      #print("NO SUCCESS:Actual:{} Predicted:{}".format(actual, result))
      return float(predictions[actual] * (100) * l_2_dist(x_clean, x_adv))
      #return 1
    #return float(result) # * l_2_dist(x_clean, x_adv))
    #return np.log(predictions[np.argmax(y_clean)]) #* l_2_dist(x_clean, x_adv))

  def l_2_constraint(x):
    return l_2_dist(x_clean, x) < max_l_2

  def minimize_l_2(z):
    #logger.to_file(f'x_adv shape in minimize_l_2 {x_adv.shape}')
    nonlocal x_adv
    x_adv_l2 = process_digit(x_adv, z.ravel(), epsilon)
    fitness = float(l_2_dist(x_clean.ravel(),x_adv_l2.ravel()))
    #adv_pred = np.argmax(model.predict(x_adv_l2.reshape((1,28,28,1))))
    #logger.to_file(f'fitness in minimize_l_2 {fitness}')
    #logger.to_file(f'Prediction in minimize_l_2 {adv_pred}')
    #np.savetxt('x_adv_l_2.csv', x_adv_l2.reshape((1, 784)), delimiter=',')
    return fitness

  @counter
  def inequality_contraint(x):
    x_adv = process_digit(x_clean, x.ravel(), epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    result = np.argmax(predictions)
    return result != np.argmax(y_clean)

  # Number of agents and decision variables
  n_agents = agents
  n_variables = 784
  # Lower and upper bounds (has to be the same size as `n_variables`)
  lower_bound = np.empty(n_variables)
  lower_bound.fill(0)
  upper_bound = np.empty(n_variables)
  upper_bound.fill(1)

  space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
  function = Function(evaluate_acc)
  #function = ConstrainedFunction(evaluate_acc, [l_2_constraint], 10000.0)

  # Bundles every piece into Opytimizer class
  opt = Opytimizer(space, optimizer, function, save_agents=False)
  #Runs the optimization task
  opt.start(n_iterations = iterations)

  xopt = opt.space.best_agent.position
  #logger.info('xopt shape: %s', xopt.shape)
  #x_adv = x_clean.ravel() + xopt.ravel() * epsilon
  #x_adv = np.array(xopt.value)
  x_adv = process_digit(x_clean, xopt.ravel(), epsilon)
  x_adv = x_adv.reshape((28,28,1))
  dist = l_2_dist(x_clean, x_adv)
  eval_count += iterations
  adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
  attack_succ = np.argmax(y_clean) != adv_pred
  logger.info(f"Attack result:{attack_succ}, Queries: {eval_count} Dist:{dist}")
  logger.to_file(f"Attack result:{attack_succ}, Queries: {eval_count} Dist:{dist}")
  return x_adv, eval_count, dist


def get_opyt_adv(model, x_test_random, y_test_random,
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
    logger.info(f"Generating example:{i}")
    #Creates the optimizer
    params={'model':model, 'x_clean':x_test_random[i], 'x_adv': None,
    'y_clean': y_test_random[i],
    'epsilon' : epsilon, 'l_2_step': l_2_step, 'l_2_min':False}
    optimizer = opytimizer.optimizers.misc.MODAOA(params=params)
    #optimizer = opytimizer.optimizers.misc.AOA()
    adv_nvg[i], count, dist = get_adv_opyt_example(model, optimizer,
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
  #l_2 = get_dataset_l_2_dist(x_test_random, x_test_nvg)
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

def get_nvg_adv(model, x_test_random, y_test_random,
                iterations = 100, epsilon = 3.55, max_l_2=6):
  iteration = round(iterations)
  print("Iterations:{}, espilon: {} and Max-L2:{}".format(
      iteration, epsilon, max_l_2))

  no_samples = len(x_test_random)
  adv_nvg = np.empty((no_samples,28,28,1))

  i = 0
  query_count = []
  l_2 = []
  params = ng.p.Array(shape=(784,)).set_bounds(0, 1)
  for i in range(no_samples):
    print("Generating example: ", i)
    params = ng.p.Array(shape=(784,)).set_bounds(0, 1)
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
           process_digit(x_test_random[i], x, epsilon), x_test_random[i]) < max_l_2)

    adv_nvg[i], count, dist = get_adv_nvg_example(model, optimizer,
                                                  x_test_random[i],
                                                  y_test_random[i] ,
                                                  epsilon = epsilon
                                                  )
    query_count.append(count)
    l_2.append(dist)

  x_test_nvg = adv_nvg
  y_pred_nvg = model.predict(x_test_nvg)
  acc = get_accuracy(y_pred_nvg, y_test_random)
  #l_2 = get_dataset_l_2_dist(x_test_random, x_test_nvg)
  l_2_mean = np.mean(l_2)
  query_mean = np.mean(query_count)
  print("\nTotal Examples: {}, Iterations:{}, espilon: {} and Max-L2:{} \nAccuracy: {} Mean L2 Counted: {} Query: {}".format(
      len(y_test_random), iterations, epsilon, max_l_2, acc, l_2_mean,query_mean,
      l_2_dist(x_test_random.ravel(), x_test_nvg.ravel())))

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
                        epsilon = 0.5):
  @counter
  def evaluate_acc(x):
    x_adv = process_digit(x_clean, x, epsilon)
    predictions = model.predict(x_adv.reshape((1,28,28,1)))[0]
    return np.log(predictions[np.argmax(y_clean)]) # * l_2_dist(x_clean, x_adv))
    #return np.log(predictions[np.argmax(y_clean)]) #* l_2_dist(x_clean, x_adv))

  xopt = optimizer.minimize(evaluate_acc)
  x = np.array(xopt.value)
  x_adv = process_digit(x_clean, x, epsilon)
  dist = l_2_dist(x_clean, x_adv)
  adv_pred = np.argmax(model.predict(x_adv.reshape((1,28,28,1))))
  attack_succ = np.argmax(y_clean) != adv_pred
  print("Attack result:{}, Queries: {} Dist:{}".format(attack_succ,
                                                       evaluate_acc.count, dist))
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
