import numpy

import torch
from torch import autograd


PYTORCH_TYPES = (autograd.Variable, torch.Tensor)


def variance(outputs_expectation_of_products, outputs_expectations):
    output_product_of_expectations = (
        outputs_expectations * outputs_expectations)
    return outputs_expectation_of_products - output_product_of_expectations


def covariance(outputs_expectation_of_products, outputs_expectations):
    output_product_of_expectations = (
        outer(outputs_expectations, outputs_expectations))

    return outputs_expectation_of_products - output_product_of_expectations


def outer(vector, other_vector):
    if isinstance(vector, PYTORCH_TYPES):
        return torch.ger(vector, other_vector)
    else:
        return numpy.outer(vector, other_vector)


def centered_covariance(outputs):
    if isinstance(outputs, PYTORCH_TYPES):
        return torch.mm(outputs.t(), outputs)
    else:
        return numpy.dot(outputs.T, outputs)


def eig(tensor):
    if isinstance(tensor, PYTORCH_TYPES):
        u, V = torch.eig(tensor, True)
        u = u[:, 0]  # remove the imaginary part
    else:
        u, V = numpy.linalg.eigh(tensor)

    return u, V


def svd(tensor, transpose=False):
    """
    NOTE
    ----
        By default numpy returns transposed V (like in svd equations) while
        torch returns non transposed V. This helper function standardize them
        returning transposed V if transpose=True and non transposed ones
        otherwise.
    """
    if isinstance(tensor, PYTORCH_TYPES):
        U, S, V = torch.svd(tensor)
        if transpose:
            V = V.t()
        return U, S, V
    else:
        U, S, V = numpy.linalg.svd(tensor)
        if not transpose:
            V = V.T
        return U, S, V


def dot(tensor, other_tensor):
    if isinstance(tensor, PYTORCH_TYPES):
        return torch.mm(tensor, other_tensor)
    else:
        return numpy.dot(tensor, other_tensor)


def expectations(outputs):
    return outputs.sum(0)


def unsqueeze(tensor, axis):
    tensor_shape = list(shape(tensor))
    tensor_shape.insert(axis, 1)
    return reshape(tensor, tensor_shape)


def concatenate(tensors, axis=0):
    if isinstance(tensors[0], PYTORCH_TYPES):
        return torch.cat(tensors, dim=axis)
    else:
        return numpy.concatenate(tensors, axis=axis)


def abs(tensor):
    if isinstance(tensor, PYTORCH_TYPES):
        return tensor.abs()
    else:
        return numpy.abs(tensor)


def reshape(tensor, *new_shape):
    try:
        return tensor.view(*new_shape)
    except TypeError:
        return tensor.reshape(*new_shape)


def shape(tensor):
    try:
        return tensor.size()
    except TypeError:
        return tensor.shape


def transpose(tensor):
    try:
        return tensor.t()
    except TypeError:
        return tensor.T


def zeros(shape, container_type):
    if isinstance(container_type, autograd.Variable):
        return torch.zeros(*shape).type(container_type.type())
    else:
        return numpy.zeros(shape)


def diag(tensor):
    if isinstance(tensor, PYTORCH_TYPES):
        return torch.diag(tensor)
    else:
        return numpy.diag(tensor)


def sqrt(tensor):
    if isinstance(tensor, PYTORCH_TYPES):
        return torch.sqrt(tensor)
    else:
        return numpy.sqrt(tensor)


def cast(tensor, tensor_precision):
    if isinstance(tensor, PYTORCH_TYPES):
        return tensor.type(type(tensor_precision))
    else:
        return tensor.astype(tensor_precision.dtype)


def convert_to_array(value, tensor_type):
    if isinstance(tensor_type, PYTORCH_TYPES):
        # TODO make it flexible
        return torch.cuda.LongTensor(value)
    else:
        return numpy.array(value)


def get_limits(dimensions, tensors):
    list_index = [i for i, item in enumerate(dimensions)
                  if isinstance(item, list)]
    if list_index:
        return None

    indexes = dimensions

    limits = []

    start = 0
    seen = 0
    for tensor in tensors:
        size = numpy.prod(shape(tensor))
        end = (start +
               (indexes[start:] < seen + size).sum())

        if start < end:
            limits.append(end - start)

        seen += size
        start += end

        if start >= len(indexes):
            break

    if limits[-1] < len(indexes):
        limits.append(len(indexes))

    return limits


def extract_subset_from_tensors(tensors, indexes):

    indexes = convert_to_array(indexes, tensors[0])
    tensor_subset = zeros([len(indexes)], tensors[-1])

    start = 0
    seen = 0
    for tensor in tensors:
        size = numpy.prod(shape(tensor))
        end = (start +
               (indexes[start:] < seen + size).sum())

        if start < end:
            tensor_indexes = indexes[start:end] - seen
            tensor_subset[start:end] = reshape(tensor, -1)[tensor_indexes]

        seen += size
        start += end
        if start >= len(indexes):
            break

    return tensor_subset
