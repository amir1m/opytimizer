import opytimizer.utils.logging as l
from opytimizer.core.function import Function

logger = l.get_logger(__name__)


class Internal(Function):
    """An Internal class, inherited from Function.
    It will server as the basis class for holding in-code related
    objective functions.

    Methods:
        _build(function): Sets an external function point to a class
        attribute.

    """

    def __init__(self, pointer=None):
        """Initialization method.

        Args:
            function (*func): This should be a pointer to a function that will
            return the fitness value.

        """

        logger.info('Overriding class: Function -> Internal.')

        # Overrides parent class with its own type
        super(Internal, self).__init__(function_type='internal')

        # Now, we need to build this class up
        self._build(pointer)

        logger.info('Class overrided.')

    def _build(self, pointer):
        """This method will serve as the object building process.
        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            function (*func): This should be a pointer to a function that will
            return the fitness value.

        """

        logger.debug('Running private method: build().')

        # We apply to class pointer's the desired function
        if pointer:
            self.pointer = pointer
        else:
            e = f"Property 'pointer' cannot be {pointer}."
            logger.error(e)
            raise RuntimeError(e)

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Type: {self.type} | Pointer: {self.pointer} | Built: {self.built}')