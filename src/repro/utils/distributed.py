from multiprocessing import Pool as PythonPool
from typing import Tuple, Any, Callable


class AbstractPool:

    def map(self, fun, args):
        raise NotImplementedError

    def starmap(self, fun, args):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def join(self):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError


class MahlerPool(AbstractPool):
    def __init__(self):
        pass

    def map(self, fun, args):
        pass

    def starmap(self, fun, args):
        pass

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


def _select_pool_ctor(item: Tuple[str, Any]) -> bool:
    key, _ = item
    return 'Pool' in key and 'Abstract' not in key


pool_impl = {
    'mahler': MahlerPool,
    'python': PythonPool
}


def make_pool(backend: str, *args, **kwargs) -> AbstractPool:
    if backend not in pool_impl:
        raise KeyError('backend: {} not found (choices: {})'.format(backend, ','.join(pool_impl.keys())))

    return pool_impl[backend](*args, **kwargs)


class LazyInstantiator:
    """ Delay instantiation or call of a function

        This is necessary in the case where an object is not pickle-able and can not be
        transferred across processes. In such cases the object creation can be delayed
        and created in each process when needed
    """
    def __init__(self, fun: Callable, *args, **kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fun(*self.args, **self.kwargs)


if __name__ == '__main__':

    print(make_pool('python', 5))
    print(make_pool('mahler'))

    def add(a, b):
        return a + b, a


    pool = make_pool('python', 5)

    ret = pool.starmap(add, [(1, 2), (3, 4)])

    print(ret)
