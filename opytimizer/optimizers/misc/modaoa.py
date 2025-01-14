"""Modified Arithmetic Optimization Algorithm.
"""

import opytimizer.math.random as r
import opytimizer.math.distribution as d
#d.generate_levy_distribution(0.09)
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer
from scipy.stats import norm
#import sys
#sys.path.append('../adv-ml/')
#from attack_utils import

from copy import deepcopy
from attack import *
from attack_utils import *
import numpy as np

logger = l.get_logger(__name__)


class MODAOA(Optimizer):
    """An MODAOA class, inherited from Optimizer.

    This is the designed class to define MODAOA-related
    variables and methods.

    References:
        L. Abualigah et al. The Arithmetic Optimization Algorithm.
        Computer Methods in Applied Mechanics and Engineering (2021).

    """

    def __init__(self,params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MODAOA.')
        l#ogger.info("Params is:%s",params)
        self.model = params['model']
        self.x_clean = deepcopy(params['x_clean'])
        self.y_clean = np.argmax(params['y_clean'])
        self.epsilon = params['epsilon']
        self.l_2_min = params['l_2_min']
        if 'dim' in params:
            self.dim = params['dim']
        else:
            self.dim = (1,28,28,1)

        logger.to_file('Clean Label:%s', self.y_clean )
        if(self.l_2_min == True):
            logger.to_file('L2_Min:%s', self.l_2_min )

        # Overrides its parent class with the receiving params
        super(MODAOA, self).__init__()

        # Minimum accelerated function
        self.a_min = 0.2

        # Maximum accelerated function
        self.a_max = 1.0

        # Sensitive parameter
        self.alpha = 5.0

        # Control parameter
        self.mu = 0.499

        # Brownin parameters
        self.delta = 0.25 # 0.25, 0.75, 0.5
        self.dt = 0.1 # 0.1,0.01, 0.01

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def a_min(self):
        """float: Minimum accelerated function.

        """

        return self._a_min

    @a_min.setter
    def a_min(self, a_min):
        if not isinstance(a_min, (float, int)):
            raise e.TypeError('`a_min` should be a float or integer')
        if a_min < 0:
            raise e.ValueError('`a_min` should be >= 0')

        self._a_min = a_min

    @property
    def a_max(self):
        """float: Maximum accelerated function.

        """

        return self._a_max

    @a_max.setter
    def a_max(self, a_max):
        if not isinstance(a_max, (float, int)):
            raise e.TypeError('`a_max` should be a float or integer')
        if a_max < 0:
            raise e.ValueError('`a_max` should be >= 0')
        if a_max < self.a_min:
            raise e.ValueError('`a_max` should be >= `a_min`')

        self._a_max = a_max

    @property
    def alpha(self):
        """float: Sensitive parameter.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0:
            raise e.ValueError('`alpha` should be >= 0')

        self._alpha = alpha

    @property
    def mu(self):
        """float: Control parameter.

        """

        return self._mu

    @mu.setter
    def mu(self, mu):
        if not isinstance(mu, (float, int)):
            raise e.TypeError('`mu` should be a float or integer')
        if mu < 0:
            raise e.ValueError('`mu` should be >= 0')

        self._mu = mu

    def update(self, space, iteration, n_iterations):
        """Wraps Arithmetic Optimization Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates math optimizer accelarated coefficient (eq. 2)
        MOA = self.a_min + iteration * ((self.a_max - self.a_min) / n_iterations)

        # Calculates math optimizer probability (eq. 4)
        MOP = 1 - (iteration ** (1 / self.alpha) / n_iterations ** (1 / self.alpha))

        #logger.info('Shape of Best Agent: %s', space.best_agent.position.shape)
        # x_adv = process_digit(self.x_clean, space.best_agent.position.ravel(), self.epsilon, dim=self.dim)
        # pred = self.model.predict(x_adv.reshape(self.dim))
        # y_pred = np.argmax(pred)
        #logger.to_file("Predicted: %s", y_pred)
        #y_true = np.argmax(self.y_clean)
        #c_l_2_dist = l_2_dist(self.x_clean.ravel(), x_adv)
        # logger.to_file("Clean Label: %s", np.argmax(self.y_clean))
        #logger.to_file("l_2_dist: %s", c_l_2_dist)
        #logger.to_file("MOP: %s",MOP)
        #logger.to_file("MOA: %s",MOA)

        # Iterates through all agents
        for agent in space.agents:
            # Iterates through all variables
            for j in range(agent.n_variables):
                # Generates random probability
                r1 = r.generate_uniform_random_number()

                # Calculates the search partition
                search_partition = (agent.ub[j] - agent.lb[j]) * self.mu + agent.lb[j]

                # If probability is bigger than MOA
                #if y_pred == y_true:
                if r1 > MOA:
                    # Generates an extra probability
                    r2 = r.generate_uniform_random_number()

                    # If probability is bigger than 0.5
                    if r2 > 0.5:
                        # Updates position with (eq. 3 - top)
                        #agent.position[j] = space.best_agent.position[j] / (MOP+ c.EPSILON)  * search_partition
                        agent.position[j] = space.best_agent.position[j] / (norm.rvs(scale=self.delta**2)+c.EPSILON) * search_partition

                    # If probability is smaller than 0.5
                    else:
                        # Updates position with (eq. 3 - bottom)
                        #agent.position[j] = space.best_agent.position[j] * (MOP + c.EPSILON) * search_partition
                        agent.position[j] = space.best_agent.position[j] * (norm.rvs(scale=self.delta**2)+c.EPSILON) * search_partition

                # If probability is smaller than MOA
                else:
                    # Generates an extra probability
                    r3 = r.generate_uniform_random_number()

                    # If probability is bigger than 0.5
                    if r3 > 0.5:
                        # Updates position with (eq. 5 - top)
                        #agent.position[j] = space.best_agent.position[j] - (MOP * search_partition)
                        #agent.position[j] = space.best_agent.position[j] - r.generate_gaussian_random_number(mean=space.best_agent.position[j], variance=10*MOP)* search_partition
                        #agent.position[j] = space.best_agent.position[j] - norm.rvs(scale=self.delta**2*search_partition)
                        agent.position[j] = space.best_agent.position[j] - norm.rvs(scale=self.delta**2)*search_partition

                    # If probability is smaller than 0.5
                    else:
                        # Updates position with (eq. 5 - bottom)
                        #agent.position[j] = space.best_agent.position[j] + (MOP * search_partition)
                        #agent.position[j] = space.best_agent.position[j] + r.generate_gaussian_random_number(mean=space.best_agent.position[j], variance=10*MOP) * search_partition
                        #agent.position[j] = space.best_agent.position[j] + norm.rvs(scale=self.delta**2*search_partition)
                        agent.position[j] = space.best_agent.position[j] + norm.rvs(scale=self.delta**2)*search_partition
