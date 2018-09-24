from collections import OrderedDict
import copy
import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def build(movement_samples):
    return ComputeMovement(movement_samples)


class ComputeMovement(object):
    def __init__(self, movement_samples):
        self.movement_samples = movement_samples

    def stack_parameters(self, parameters):
        stacked_parameters = {}
        for key in parameters.keys():
            stacked_parameters[key] = torch.stack(parameters[key])

        return stacked_parameters

    def compute_parameter_diff(self, diffs, new_parameters, reference_parameters):

        if 'all_parameters' in reference_parameters:
            all_new_parameters = torch.cat(tuple(m.view(-1) for m in new_parameters.values()))

            if 'all_parameters' not in diffs:
                diffs['all_parameters'] = []

            diffs['all_parameters'].append(all_new_parameters -
                                           reference_parameters['all_parameters'])

        keys = set()
        for key in new_parameters.keys():
            if key not in diffs:
                diffs[key] = []

            diffs[key].append(new_parameters[key].view(-1) - reference_parameters[key].view(-1))
            keys.add(key)

        if list(set(reference_parameters.keys()) - keys) not in [[], ['all_parameters']]:
            remaining_keys = set(reference_parameters.keys()) - keys
            raise RuntimeError(
                "Keys {} are missing in new parameters dict".format(remaining_keys))

    def compute_movement(self, reference, batch_out_idx, model, device, analysis_loader):
        # TODO: The diff vectors are not exactly on the same data. Is it problematic?
        # They are not on the same data because each diff vector has a different batch_out_idx
        diffs = []
        n_samples = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(analysis_loader):

                # Commented out because infinit sampler will sample many mini-batches with similar
                # examples, thus it won't be possible anymore to evaluate only on non-seen examples,
                # unless we use another set.

                # if batch_idx == batch_out_idx:
                #     continue

                data, target = data.to(device), target.to(device)
                softmax = F.softmax(self.model(data))
                diffs.append((softmax - reference[batch_idx]).view(-1))

                n_samples += diffs[-1].size(0)
                if n_samples >= self.movement_samples:
                    break

        # diff = torch.cat(diffs)
        # sqr_diff = diff * diff
        # return sqr_diff.mean(0)
        return torch.cat(diffs)[:self.movement_samples]

    def __call__(self, results, name, set_name, analysis_loader, training_loader, model, optimizer,
                 device):

        if "function" not in results or "_references" not in results['function']:
            raise RuntimeError("Cannot compute movement without computing reference analysis "
                               "beforehand.")

        parameters = OrderedDict()
        movements = []
        original_state = copy.deepcopy(model.state_dict())

        reference_function = results['function'].pop('_references')
        reference_parameters = results['parameters'].pop('_references')

        for batch_idx, (data, target) in enumerate(training_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            model.load_state_dict(original_state)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.compute_parameter_diff(
                    parameters, OrderedDict(model.named_parameters()), reference_parameters)
                movements.append(
                    self.compute_movement(reference_function, batch_idx, model, device, analysis_loader))

        logger.info("Freing reference from GPU mem")
        del reference_function
        torch.cuda.empty_cache()

        with torch.no_grad():
            parameters = self.stack_parameters(parameters)
        logger.info("Freing parameters from GPU mem")
        torch.cuda.empty_cache()

        with torch.no_grad():
            movements = torch.stack(movements)
        logger.info("Freing movements from GPU mem")
        torch.cuda.empty_cache()

        model.load_state_dict(original_state)

        return {'parameters': {'_movement': parameters},
                'function': {'_movement': movements}}
