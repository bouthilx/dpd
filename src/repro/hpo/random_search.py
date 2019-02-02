import random

from mahler.core.utils.flatten import unflatten


def build(space):
    return RandomSearch(space)


class RandomSearch(object):
    def __init__(self, space):
        self.space = space

    def observe(self, trials):
        pass

    def get_params(self):
        randomseed = random.randint(0, 100000)
        arguments = unflatten(dict(zip(self.space.keys(), self.space.sample(seed=randomseed)[0])))
        logger.info('Sampling:\n{}'.format(pprint.pformat(arguments)))
        return arguments
