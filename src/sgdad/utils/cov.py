import logging
import numpy

import torch
from torchnet.meter.meter import Meter

from . import backend


def to_np(x):
    if isinstance(x, numpy.ndarray):
        return x
    if hasattr(x, 'data'):
        x = x.data
    return x.cpu().numpy()


logger = logging.getLogger(__name__)

dtype = torch.FloatTensor


class ExpectationMeter(Meter):
    """
    Parameters
    ----------
    method: string
        either "kahan" or "neumaier". Anything else will result in the naive
        behavior which is unstable numerically.
    """
    def __init__(self, method="kahan"):
        self.method = method
        self.reset()

    def initialize(self, value):
        n_dimensions = backend.shape(value)[1:]
        self.sum_of_values = backend.zeros(n_dimensions, value)
        self._numerical_error_buffer = backend.zeros(n_dimensions, value)

    def reset(self):
        self.sum_of_values = None
        self._numerical_error_buffer = None
        self.n = 0.

    def add(self, value, n=1):
        if self.sum_of_values is None:
            self.initialize(value)

        # https://en.wikipedia.org/wiki/Kahan_summation_algorithm#The_algorithm
        if self.method == "kahan":
            # Remove buffered error from value
            value = (backend.expectations(value) -
                     self._numerical_error_buffer)

            # Compute unbalanced sum
            sum_of_values = self.sum_of_values + value

            # Recovers the missing precision
            self._numerical_error_buffer = (
                (sum_of_values - self.sum_of_values) - value)

            self.sum_of_values = sum_of_values

        # https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements
        elif self.method == "neumaier":
            # Remove buffered error from value
            value = backend.expectations(value)
            sum_of_values = self.sum_of_values + value

            use_sum = backend.abs(self.sum_of_values) >= backend.abs(value)
            use_sum = backend.cast(use_sum, value)

            # if sum >= value
            self._numerical_error_buffer += (
                use_sum * ((self.sum_of_values - sum_of_values) + value))
            # else value > sum
            self._numerical_error_buffer += (
                (1 - use_sum) * ((self.sum_of_values - sum_of_values) + value))

            self.sum_of_values = sum_of_values
        else:
            self.sum_of_values += backend.expectations(value)

        self.n += n

    def value(self):
        if self.method == "neumaier":
            return (self.sum_of_values + self._numerical_error_buffer) / self.n
        else:
            return self.sum_of_values / self.n


class CovarianceMeter(Meter):
    def __init__(self, centered=False, **kwargs):
        super(CovarianceMeter, self).__init__()
        self.centered = centered
        self.expectation_of_products_meter = ExpectationMeter(**kwargs)
        self.expectation_meter = ExpectationMeter(**kwargs)
        self.reset()

    @property
    def n(self):
        return self.expectation_of_products_meter.n + 1

    def initialize(self, value):
        self.expectation_of_products_meter.initialize(
            backend.unsqueeze(backend.centered_covariance(value), 0))
        self.expectation_meter.initialize(value)

    def reset(self):
        self.expectation_of_products_meter.reset()
        self.expectation_meter.reset()
        self.expectation_of_products_meter.n = -1
        self.expectation_meter.n = -1

    def add(self, value, n=1):

        cov = backend.centered_covariance(value)

        self.expectation_of_products_meter.add(
            backend.unsqueeze(cov, 0), n=n)

        if not self.centered:
            self.expectation_meter.add(value, n=n)

    def value(self):
        if not self.centered:
            return backend.covariance(
                self.expectation_of_products_meter.value(),
                self.expectation_meter.value())
        else:
            return self.expectation_of_products_meter.value()


class CorrelationMeter(CovarianceMeter):
    def __init__(self, centered=False, epsilon=1e-8):
        self.epsilon = epsilon
        super(CorrelationMeter, self).__init__(centered)

    def value(self):
        cov = super(CorrelationMeter, self).value()
        std = backend.sqrt(backend.diag(cov))
        return cov / (backend.outer(std, std) + self.epsilon)


class ApproximateMeter(Meter):
    def __init__(self, meter, n_dimensions, axis=None, sticky_dimensions=True, **kwargs):

        self.meter = meter
        self.n_dimensions = n_dimensions
        self.axis = axis
        self.sticky_dimensions = sticky_dimensions
        self.reset()
        super(ApproximateMeter, self).__init__(**kwargs)

    @property
    def n(self):
        return self.meter.n

    def initialize(self, value=None, n_dimensions=None):
        if n_dimensions is None and self.dimensions is None:
            if self.axis is None:
                n_dimensions = numpy.prod(backend.shape(value)[1:])
            else:
                n_dimensions = backend.shape(value)[self.axis + 1]

        if self.dimensions is None:
            # if n_dimensions < self.n_dimensions:
            #     logger.debug("Using %d rather than %d" %
            #                  (n_dimensions, self.n_dimensions))
            #     size = n_dimensions
            # else:
            #     size = self.n_dimensions
            size = n_dimensions

            self.dimensions = self.sample_dimensions(value, size, self.axis)

    def sample_dimensions(self, value, size, axis):
        dimensions = list(numpy.sort(numpy.random.choice(
            numpy.arange(size),  # self.n_dimensions),
            size=self.n_dimensions,
            replace=False)))

        if axis is not None:
            indexes = [numpy.random.randint(dim_size)
                       for dim_size in backend.shape(value)[1:]]
            indexes[self.axis] = dimensions
            dimensions = indexes

        return dimensions

    def reset(self):
        if (not self.sticky_dimensions or
                getattr(self, "dimensions", None) is None):
            self.dimensions = None

        self.meter.reset()

    def get_subset(self, value):

        if self.axis is None:
            batch_size = backend.shape(value)[0]
            return backend.reshape(value, (batch_size, -1))[:, self.dimensions]
        else:
            return to_np(value)[[slice(None)] + self.dimensions]

    def add(self, value, n=1):
        if self.dimensions is None:
            self.initialize(value)

        self.meter.add(self.get_subset(value), n)

    def value(self):
        return self.meter.value()
