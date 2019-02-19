import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent

logger = l.get_logger(__name__)


class Space:
    """A Space class that will hold agents, variables and methods
    related to the search space.

    Properties:
        n_agents (int): Number of agents.
        n_variables (int): Number of decision variables.
        n_dimensions (int): Dimension of search space.
        n_iterations (int): Number of iterations.
        agents (list): List of agents that belongs to Space.
        best_agent (Agent): A best agent object from Agent class.
        lb (np.array): Lower bound array with the minimum possible values.
        ub (np.array): Upper bound array with the maximum possible values.
        built (boolean): A boolean to indicate whether the space is built.

    Methods:
        _check_bound_size(bound, size): Checks whether the bound's length
        is equal to size parameter.
        _create_agents(n_agents, n_variables, n_dimensions): Creates a list of agents.
        _initialize_agents(agents, lower_bound, upper_bound): Initialize the Space's agents,
        setting random numbers to their position.
        _build(lower_bound, upper_bound): An object building method.

    """

    def __init__(self, n_agents=1, n_variables=2, n_dimensions=1, n_iterations=10, lower_bound=None, upper_bound=None):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.
            n_iterations (int): Number of iterations.
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.info('Creating class: Space.')

        # Number of agents
        self._n_agents = n_agents

        # Number of variables
        self._n_variables = n_variables

        # Number of dimensions
        self._n_dimensions = n_dimensions

        # Number of iterations
        self._n_iterations = n_iterations

        # Agent's list
        self._agents = None

        # Best agent object
        self._best_agent = None

        # Lower and upper bounds
        self._lb = np.zeros(n_variables)
        self._ub = np.ones(n_variables)

        # Indicates whether the space is built or not
        self._built = False

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # We will log some important information
        logger.info('Class created.')

    @property
    def n_agents(self):
        """Number of agents.
        """

        return self._n_agents

    @property
    def n_variables(self):
        """Number of decision variables.
        """

        return self._n_variables

    @property
    def n_dimensions(self):
        """Dimension of search space.
        """

        return self._n_dimensions

    @property
    def n_iterations(self):
        """Number of iterations.
        """

        return self._n_iterations

    @property
    def agents(self):
        """List of agents that belongs to Space.
        """

        return self._agents

    @agents.setter
    def agents(self, agents):
        self._agents = agents

    @property
    def best_agent(self):
        """A best agent object from Agent class.
        """

        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent):
        self._best_agent = best_agent

    @property
    def lb(self):
        """Lower bound array with the minimum possible values.
        """

        return self._lb

    @lb.setter
    def lb(self, lb):
        self._lb = lb

    @property
    def ub(self):
        """Upper bound array with the maximum possible values.
        """

        return self._ub

    @ub.setter
    def ub(self, ub):
        self._ub = ub

    @property
    def built(self):
        """A boolean to indicate whether the space is built.
        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _check_bound_size(self, bound, size):
        """Checks if the bounds' size are the same of
        variables size.

        Args:
            bound(np.array): bounds array.
            size(int): size to be checked.

        Returns:
            True if sizes are equal.

        """

        logger.debug('Running private method: check_bound_size().')

        if len(bound) != size:
            e = f'Expected size is {size}. Got {len(bound)}.'
            logger.error(e)
            raise RuntimeError(e)
        else:
            logger.debug('Bound checked.')
            return True

    def _create_agents(self, n_agents, n_variables, n_dimensions):
        """Creates and populates the agents array.
        Also defines a random best agent, only for initialization purposes.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.

        Returns:
            A list of agents and a best agent.

        """

        logger.debug('Running private method: create_agents().')

        # Creating an agents list
        agents = []

        # Iterate through number of agents
        for _ in range(n_agents):
            # Appends new agent to list
            agents.append(
                Agent(n_variables=n_variables, n_dimensions=n_dimensions))

        # Apply a random agent as the best one
        best_agent = agents[0]

        return agents, best_agent

    def _initialize_agents(self, agents, lower_bound, upper_bound):
        """Initialize agents' position array with
        uniform random numbers.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterate through number of agents
        for agent in agents:
            # For every variable
            for var in range(agent.n_variables):
                # we generate uniform random numbers
                agent.position[var] = r.generate_uniform_random_number(
                    lower_bound[var], upper_bound[var], size=agent._n_dimensions)

    def _build(self, lower_bound, upper_bound):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.debug('Running private method: build().')

        # Checking if lower bound is avaliable
        if lower_bound:
            # Check if its size matches to our actual number of variables
            if self._check_bound_size(lower_bound, self.n_variables):
                self.lb = lower_bound
        else:
            e = f"Property 'lower_bound' cannot be {lower_bound}."
            logger.error(e)
            raise RuntimeError(e)

        # We need to check upper bounds as well
        if upper_bound:
            if self._check_bound_size(upper_bound, self.n_variables):
                self.ub = upper_bound
        else:
            e = f"Property 'upper_bound' cannot be {upper_bound}."
            logger.error(e)
            raise RuntimeError(e)

        # Creating agents
        self.agents, self.best_agent = self._create_agents(
            self.n_agents, self.n_variables, self.n_dimensions)

        # Initializing agents
        self._initialize_agents(self.agents, self.lb, self.ub)

        # If no errors were shown, we can declared the Space as built
        self.built = True

        # Logging attributes
        logger.debug(
            f'Agents: {self.n_agents} | Size: ({self.n_variables}, {self.n_dimensions})'
            + f' | Iterations: {self.n_iterations} | Lower Bound: {self.lb}'
            + f' | Upper Bound: {self.ub} | Built: {self.built}')