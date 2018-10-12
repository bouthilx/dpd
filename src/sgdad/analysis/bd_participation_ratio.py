from collections import OrderedDict
import copy
import logging

import numpy

import torch
import torch.nn.functional as F

from tqdm import tqdm

from sgdad.utils.cov import ApproximateMeter, CovarianceMeter


logger = logging.getLogger(__name__)


def build(movement_samples, at_origin, center, normalize, batch_size, optimizer_params):
    return ComputeBlockDiagonalParticipationRatio(movement_samples, at_origin, center, normalize, batch_size, optimizer_params)


class ComputeBlockDiagonalParticipationRatio(object):
    def __init__(self, movement_samples, at_origin, center, normalize, batch_size, optimizer_params):
        self.movement_samples = movement_samples
        self.at_origin = at_origin
        self.center = center
        self.normalize = normalize
        self.batch_size = batch_size
        self.optimizer_params = optimizer_params

        if self.center and self.normalize:
            raise RuntimeError(
                "We cannot compute centering before normalizing. If normalize is true, "
                "we assume the data is already centered before normalizing")

    def update_parameter_covs(self):
        keys = set()
        for key, value in self.named_parameters():
            if self.at_origin:
                value -= self.reference_parameters[key]
            if self.normalize:
                value /= value.norm()
            self.parameter_Cs[key].add(value.view(1, -1), n=1)
            # - reference_parameters[key].view(-1))
            keys.add(key)

        if list(set(self.reference_parameters.keys()) - keys):
            remaining_keys = set(self.reference_parameters.keys()) - keys
            raise RuntimeError(
                "Keys {} are missing in new parameters dict".format(remaining_keys))

    def compute_movement(self, batch_out_idx):
        # TODO: The diff vectors are not exactly on the same data. Is it problematic?
        # They are not on the same data because each diff vector has a different batch_out_idx
        diffs = []
        n_samples = 0
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.analysis_loader):

                # Commented out because infinit sampler will sample many mini-batches with similar
                # examples, thus it won't be possible anymore to evaluate only on non-seen examples,
                # unless we use another set.

                # if batch_idx == batch_out_idx:
                #     continue

                data, target = data.to(self.device), target.to(self.device)
                softmax = F.softmax(self.model(data))
                diffs.append(softmax - self.reference_function[batch_idx])

                n_samples += data.size(0)
                if n_samples >= self.movement_samples:
                    break

        return torch.cat(diffs)[:self.movement_samples].view(-1)

    def update_function_cov(self, batch_out_idx):
        movement = self.compute_movement(batch_out_idx)
        if self.normalize:
            movement /= movement.norm()
        self.function_C.add(movement.unsqueeze(0), n=1)

    def compute_references(self):
        self.compute_reference_parameters()
        self.compute_reference_function()

    def compute_reference_function(self):
        references = []
        n_samples = 0
        self.model.eval()

        for batch_idx, (data, target) in enumerate(self.analysis_loader):
            data = data.to(self.device)
            references.append(F.softmax(self.model(data)))
            n_samples += data.size(0)
            if n_samples >= self.movement_samples:
                break

        # shape is (batch_idx, sample_idx, dimension_idx)
        self.reference_function = torch.stack(references)

    def named_parameters(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'BatchNorm2d':
                w = copy.deepcopy(module.weight.data).unsqueeze(1)
            elif module.__class__.__name__ == 'Conv2d':
                w = copy.deepcopy(module.weight.data).view(module.weight.shape[0], -1).contiguous()
            elif module.__class__.__name__ == 'Linear':
                w = copy.deepcopy(module.weight.data)
            else:
                continue

            if getattr(module, 'bias', None) is not None:
                w = torch.cat([w, module.bias.unsqueeze(1)], dim=1)

            yield name, w

    def compute_reference_parameters(self):
        self.reference_parameters = OrderedDict()

        self.reference_parameters.update(
            OrderedDict((key, copy.deepcopy(value.detach()))
                        for key, value in self.named_parameters()))

    def initialize_covs(self):
        self.parameter_Cs = OrderedDict()
        for key, value in self.named_parameters():
            size = numpy.prod(value.size())
            cov_meter = ApproximateMeter(
                CovarianceMeter(centered=not self.center),
                n_dimensions=min(size, self.number_of_batches))
            self.parameter_Cs[key] = cov_meter
        # self.function_C = ApproximateMeter(CovarianceMeter(), n_dimensions=500)
        self.function_C = CovarianceMeter(centered=not self.center)

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

        for batch_idx, mini_batch in enumerate(tqdm(self.training_loader, desc='computing points')):
            self.restore_checkpoint()
            self.make_one_step(mini_batch)
            with torch.no_grad():
                self.update_covs(batch_idx)

        self.restore_checkpoint()
        self.destroy_checkpoint()

        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()

        function_pr = self.compute_function_pr()
        parameter_pr = self.compute_parameter_pr()

        delattr(self, 'parameter_Cs')
        delattr(self, 'function_C')

        logger.info("Freing GPU mem")
        torch.cuda.empty_cache()

        return function_pr, parameter_pr

    def compute_function_pr(self):
        cov = self.function_C.value()
        return (torch.diag(cov).sum() ** 2) / (cov * cov).sum()

    def compute_parameter_pr(self):
        root_enumerator = 0.0
        denominator = 0.0
        for cov_meter in self.parameter_Cs.values():
            cov = cov_meter.value()
            root_enumerator += torch.diag(cov).sum()
            denominator += (cov ** 2).sum()

        return (root_enumerator ** 2) / denominator

    def make_one_step(self, mini_batch):
        # TODO: Adjust loss to analysis mini-batch size
        #       When model has batch-norm, we should compute forward on original batch-size but
        #       compute the gradient only on the number of examples equal to analysis batch-size.
        # NOTE: What if analysis batch size is larger than the original batch-size?

        # If we want N batches, but batch-size != analysis-batch-size
        # What is important is not the number of samples, but number of steps, thus number of
        # batches. If we take analysis-batch-size in batch to compute the loss, then we are fine.
        # NOTE: Should not support analysis-batch-size > batch-size, does not make sense

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

    def update_covs(self, batch_idx):
        # self.compute_parameter_diff(
        self.update_function_cov(batch_idx)
        self.update_parameter_covs()

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

        function_pr, parameter_pr = self.main_loop()

        self.restore_optimizer_params()

        return {'parameters': {'participation_ratio': {'block_diagonal': parameter_pr.item()}},
                'function': {'participation_ratio': {'full': function_pr.item()}}}
