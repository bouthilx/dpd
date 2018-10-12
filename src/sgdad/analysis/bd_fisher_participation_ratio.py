from collections import OrderedDict
import copy
import logging

import numpy

import torch
import torch.nn.functional as F

from tqdm import tqdm

from sgdad.analysis.kfac_fisher_rao_norm import FisherRaoNormKFAC
from sgdad.utils.cov import ApproximateMeter, CovarianceMeter


logger = logging.getLogger(__name__)
input = lambda msg: None


class BatchsizeError(Exception):
    pass


def build(center, batch_size, optimizer_params):
    return ComputeBlockDiagonalParticipationRatio(center, batch_size, optimizer_params)


class ComputeBlockDiagonalParticipationRatio(object):
    def __init__(self, center, batch_size, optimizer_params):
        self.center = center
        self.batch_size = batch_size
        self.optimizer_params = optimizer_params
        self.parameter_Cs = OrderedDict()
        
        if self.center and self.normalize:
            raise RuntimeError(
                "We cannot compute centering before normalizing. If normalize is true, "
                "we assume the data is already centered before normalizing")

    def update_parameter_covs(self, batch_out_idx):
        projections = self.compute_fisher_projection(batch_out_idx)
        input("Computing projections cov")
        for param, projection in projections:
            if param not in self.parameter_Cs:
                self.initialize_cov(param, projection)
            with torch.no_grad():
                self.parameter_Cs[param].add(projection.view(1, -1), n=1)
        input("Freing GPU mem")
        del projections
        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()
        input("Freing GPU mem freed")

    def compute_fisher_projection(self, batch_out_idx):
        self.model.train()

        nbatches = 0
        for mini_batch in self.analysis_loader:
            if mini_batch[0].size(0) >= self.batch_size:
                nbatches += 1

        input("Computing new F")

        metric = FisherRaoNormKFAC(self.model)

        criterion = torch.nn.CrossEntropyLoss()

        nbatches_check = 0
        for batch_idx, mini_batch in enumerate(self.analysis_loader):
            try:
                self.compute_loss(mini_batch).backward()
                nbatches_check += 1
            except BatchsizeError:
                continue
            # Note there is no optimizer step here.
            with torch.no_grad():
                metric.update_stats(nbatches)

        projections = list(metric.get_projs())
        input("Will delete metric")
        for hook in metric.hooks:
            hook.remove()
        del metric.param_groups
        # del metric.state
        del metric
        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()
        input("GPU mem freed")

        assert nbatches_check == nbatches

        return projections

    def initialize_cov(self, key, value):
        size = numpy.prod(value.size())
        cov_meter = ApproximateMeter(
            CovarianceMeter(centered=not self.center),
            n_dimensions=min(size, self.number_of_batches))
        self.parameter_Cs[key] = cov_meter

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

        for batch_idx, mini_batch in enumerate(tqdm(self.training_loader, desc='computing points')):
            self.restore_checkpoint()
            try:
                self.make_one_step(mini_batch)
            except BatchsizeError:
                continue
            self.update_covs(batch_idx)

        self.restore_checkpoint()
        self.destroy_checkpoint()

        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()

        parameter_pr = self.compute_parameter_pr()

        delattr(self, 'parameter_Cs')
        self.parameter_Cs = OrderedDict()

        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()

        return parameter_pr

    def compute_parameter_pr(self):
        root_enumerator = 0.0
        denominator = 0.0
        for cov_meter in self.parameter_Cs.values():
            cov = cov_meter.value()
            root_enumerator += torch.diag(cov).sum()
            denominator += (cov ** 2).sum()

        return (root_enumerator ** 2) / denominator

    def compute_loss(self, mini_batch):

        # TODO: Adjust loss to analysis mini-batch size
        #       When model has batch-norm, we should compute forward on original batch-size but
        #       compute the gradient only on the number of examples equal to analysis batch-size.
        # NOTE: What if analysis batch size is larger than the original batch-size?

        # If we want N batches, but batch-size != analysis-batch-size
        # What is important is not the number of samples, but number of steps, thus number of
        # batches. If we take analysis-batch-size in batch to compute the loss, then we are fine.
        # NOTE: Should not support analysis-batch-size > batch-size, does not make sense

        # self.model.train()
        data, target = mini_batch
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if self.batch_size:
            if self.batch_size > output.size(0):
                raise BatchsizeError(
                    "Cannot compute loss over {} samples in a mini-batch of size {}".format(
                        self.batch_size, output.size(0)))
            output = output[:self.batch_size]
            target = target[:self.batch_size]
        return F.cross_entropy(output, target)

    def make_one_step(self, mini_batch):
        self.compute_loss(mini_batch).backward()
        self.optimizer.step()

    def update_covs(self, batch_idx):
        self.update_parameter_covs(batch_idx)

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

        return {'function': {'participation_ratio': {'fisher_block_diagonal': parameter_pr.item()}}}
