from collections import OrderedDict
import copy
import logging

import numpy

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm


logger = logging.getLogger(__name__)


def build(movement_samples, centered, batch_size, optimizer_params):
    return ComputeTruncatedBlockDiagonalParticipationRatio(movement_samples, centered, batch_size, optimizer_params)


class ComputeTruncatedBlockDiagonalParticipationRatio(object):
    def __init__(self, movement_samples, centered, batch_size, optimizer_params):
        self.movement_samples = movement_samples
        self.centered = centered
        self.batch_size = batch_size
        self.optimizer_params = optimizer_params

    def _apply(self, fct, *args, **kwargs):
        keys = set()
        for key, value in self.model.named_parameters():
            fct(self.parameter_Cs[key], value.view(-1), *args, **kwargs)
            # - reference_parameters[key].view(-1))
            keys.add(key)

        if list(set(self.reference_parameters.keys()) - keys):
            remaining_keys = set(self.reference_parameters.keys()) - keys
            raise RuntimeError(
                "Keys {} are missing in new parameters dict".format(remaining_keys))

    def update_random_projections(self):
        def f(cov_meter, value):
            cov_meter.compute_random_projection(value)
        self._apply(f)

    def update_subspace_projections(self):
        def f(cov_meter, value):
            cov_meter.compute_subspace_projection(value)
        self._apply(f)

    def compute_subspace_basis(self):
        def f(cov_meter, value):
            cov_meter.compute_subspace_basis()
        self._apply(f)

    def compute_references(self):
        self.compute_reference_parameters()

    def compute_reference_parameters(self):
        self.reference_parameters = OrderedDict()

        self.reference_parameters.update(
            OrderedDict((key, None)  # copy.deepcopy(value))
                        for key, value in self.model.named_parameters()))

    def initialize_covs(self):
        self.parameter_Cs = OrderedDict()
        for key, value in self.model.named_parameters():
            size = int(numpy.prod(value.size()))
            self.parameter_Cs[key] = TruncatedCovariance(
                n=self.number_of_batches, d=size, k=min(50, size), p=0)

    def compute_random_projections(self):
        desc = 'computing random projection'
        for mini_batch in tqdm(self.training_loader, desc=desc):
            self.restore_checkpoint()
            self.make_one_step(mini_batch)
            with torch.no_grad():
                self.update_random_projections()

    def compute_subspace_projections(self):
        desc = 'computing subpace projection'
        for mini_batch in tqdm(self.training_loader, desc=desc):
            self.restore_checkpoint()
            self.make_one_step(mini_batch)
            with torch.no_grad():
                self.update_subspace_projections()

    def make_original_checkpoint(self):
        self.original_model_state = copy.deepcopy(self.model.state_dict())
        self.original_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def restore_checkpoint(self):
        self.model.load_state_dict(self.original_model_state)
        self.optimizer.load_state_dict(self.original_optimizer_state)

    def destroy_checkpoint(self):
        self.original_model_state = None
        self.original_optimizer_state = None

    def main_loop(self):
        self.make_original_checkpoint()

        self.number_of_batches = len(self.training_loader)
        print("Analysing participation ratio on {} batches".format(self.number_of_batches))
        self.initialize_covs()
        self.compute_references()

        self.compute_random_projections()
        self.compute_subspace_basis()
        self.compute_subspace_projections()

        self.restore_checkpoint()
        self.destroy_checkpoint()

        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()

        parameter_pr = self.compute_parameter_pr()

        delattr(self, 'parameter_Cs')

        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()

        return parameter_pr

    def compute_parameter_pr(self):
        root_enumerator = 0.0
        denominator = 0.0
        for cov_meter in self.parameter_Cs.values():
            cov = cov_meter.compute_covariance()
            root_enumerator += torch.diag(cov).sum()
            denominator += (cov ** 2).sum()

        return (root_enumerator ** 2) / denominator

    def make_one_step(self, mini_batch):
        self.model.train()
        data, target = mini_batch
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if self.batch_size:
            if self.batch_size > output.size(0):
                raise RuntimeError(
                    "Cannot compute loss over {} samples in a mini-batch of size {}".format(
                        self.batch_size, output.size(0)))
            output = output[:self.batch_size]
            target = target[:self.batch_size]
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()

    def set_optimizer_params(self):

        if getattr(self, 'original_params', None) is not None:
            raise RuntimeError("Cannot set optimizer params if not previously restored"
                               "using self.restore_optimizer_params()")

        self.original_params = self.optimizer.param_groups

        # Don't want to deep copy the params, hyper-parameters will be overwritten anyway.
        new_params = copy.copy(self.original_params)
        for group in new_params:
            group.update(self.optimizer_params)
        self.optimizer.param_groups = new_params

    def restore_optimizer_params(self):
        if getattr(self, 'original_params', None) is None:
            raise RuntimeError("Cannot restore optimizer params if not previously set"
                               "using self.set_optimizer_params()")

        self.optimizer.param_groups = self.original_params
        self.original_params = None

    def __call__(self, results, name, set_name, analysis_loader, training_loader, model, optimizer,
                 device):
        self.model = model
        self.training_loader = training_loader
        self.analysis_loader = analysis_loader
        self.optimizer = optimizer
        self.device = device

        self.set_optimizer_params()

        parameter_pr = self.main_loop()

        self.restore_optimizer_params()

        return {'parameters.participation_ratio.sketch.block_diagonal': parameter_pr.item()}


class TruncatedCovariance(object):
    def __init__(self, n, d, k, p):
        self.n = n
        self.d = d
        self.k = k
        self.p = p
        self.mean = torch.cuda.FloatTensor(d, ).fill_(0.0)
        self.O = torch.cuda.FloatTensor(n, k + p).uniform_()
        self.Y = torch.cuda.FloatTensor(d, k + p).fill_(0.0)
        self.i = 0

    def compute_random_projection(self, row):
        # row := (1, d)
        # O := (n, k + p)
        # Y := (d, k + p)
        self.mean.add_(row)
        self.Y.addr_(row, self.O[self.i, :])
        self.i += 1
        self.verify_overshoot()

    def compute_subspace_basis(self):
        self.verify_incomplete()
        self.mean.div_(self.n)
        self.Y.addr_(alpha=-1, vec1=self.mean, vec2=self.O.sum(0))
        self.Qt = torch.qr(self.Y)[0].t()
        self.Y = None
        self.mean = None
        torch.cuda.empty_cache()
        self.B = torch.cuda.FloatTensor(self.n, self.k + self.p).fill_(0.0)
        # self.a, self.tau = torch.geqrf(self.Y)
        self.i = 0

    def compute_subspace_projection(self, row):
        self.verify_subspace_basis()
        # row := (1, d)
        # Qt := (k + p, d)
        # B := (n, k + p)
        # self.B[self.i, :] = torch.ormqr(self.a, self.tau, row.unsqueeze(0), transpose=True)
        self.B[self.i, :] = torch.mv(self.Qt, row)
        self.i += 1
        self.verify_overshoot()

    def compute_covariance(self):
        self.verify_incomplete()
        cov = torch.mm(self.B.t(), self.B)[:self.k, :self.k] / (self.n - 1)
        self.B = None
        torch.cuda.empty_cache()
        return cov

    def verify_overshoot(self):
        if self.i > self.n:
            raise RuntimeError("Expected {} examples, current one is the {}-th one.".format(
                self.n, self.i))

    def verify_incomplete(self):
        if self.i < self.n:
            raise RuntimeError("Expected {} examples, only received {} yet.".format(
                self.n, self.i))

    def verify_subspace_basis(self):
        if getattr(self, 'Qt', None) is None:
            raise RuntimeError(
                "Cannot compute subspace projection without first computing the subspace basis.")


def truncated_s(A, k, p=10):
    d, n = A.shape

    if k + p > min(d, n):
        raise ValueError('k + p must be <= min(A.shape)')
    if A.is_cuda:
        O = torch.cuda.FloatTensor(n, k + p).uniform_()
    else:
        O = torch.randn(n, k + p).cuda()

    # a_i : (1, d)

    # O := (n, k + p)

    # Y := (d, k + p)
    Y = torch.mm(A, O)
    # Q := (d, k + p)
    Q, _ = torch.qr(Y)
    # B := (k + p, n)
    B = torch.mm(Q.t(), A)
    # B := (k + p, k + p)
    BB = torch.mm(B, B.t())
    # S := (k + p)
    S = torch.symeig(BB, eigenvectors=False)
    return S[:k]


def _test(A):
    d, n = A.shape
    print(d)
    k = 64
    p = 0
    C = A - A.mean(0)
    cov = torch.mm(C, C.t())
    cov /= (n - 1)
    print(list(sorted(e.item() for e in torch.diag(cov)))[-10:])
    pr = (torch.diag(cov).sum() ** 2 / (cov * cov).sum()).item()
    print(pr)
    print(pr / cov.size(0))

    truncated_cov = TruncatedCovariance(n, d, k, p)

    for i in tqdm(range(n), total=n, desc='Random proj'):
        truncated_cov.compute_random_projection(A[:, i])

    truncated_cov.compute_subspace_basis()

    for i in tqdm(range(n), total=n, desc='Subspace proj'):
        truncated_cov.compute_subspace_projection(A[:, i])

    cov = truncated_cov.compute_covariance()
    print(cov.size())
    print(list(sorted(e.item() for e in torch.diag(cov)))[-10:])
    pr = (torch.diag(cov).sum() ** 2 / (cov * cov).sum()).item()
    print(pr)
    print(pr / cov.size(0))


if __name__ == '__main__':
    A = torch.randn(100, 1024 * 1024).cuda()
    _test(A)

    trainset = torchvision.datasets.MNIST(root='.', train=True,
                                          download=True,
                                          transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000)

    # Create the train/valid/test splits
    for d, l in trainloader:
        data, labels = d, l
    data = data.to(torch.device('cuda'))
    s = data.shape
    print(s)
    _test(data.view(s[0], -1).t())
