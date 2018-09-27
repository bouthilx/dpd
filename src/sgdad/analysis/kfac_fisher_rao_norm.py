from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer


def build():
    return ComputeFisherRaoNorm()


class FisherRaoNormKFAC(Optimizer):

    def __init__(self, net):
        """ Fisher Rao Norm using K-FAC approximation.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to compute the norm of.

        """
        self.params = []
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                if mod_class == 'Conv2d':
                    # Adding gathering filter for convolution
                    d['gathering_filter'] = self._get_gathering_filter(mod)
                self.params.append(d)
        super(FisherRaoNormKFAC, self).__init__(self.params, {})

    def update_stats(self, nbatches):
        """Updates xxt and ggt for each layer of the network.

        Args:
            nbatches (int): Total number of mini-batches.

        """
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, _ = group['params']
            else:
                weight = group['params'][0]
            state = self.state[weight]
            self._compute_covs(group, state, nbatches)

    def get_norm(self):
        """Returns <theta, F theta> for the whole network."""
        norm = 0
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Preconditionning
            norm += self._get_norm_one_module(weight, bias, group, state)
        return norm

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _get_norm_one_module(self, weight, bias, group, state):
        """Returns <theta, F theta> for one layer, using K-FAC for F."""
        xxt = state['xxt']
        ggt = state['ggt']
        w = weight
        s = w.shape
        if group['layer_type'] == 'Conv2d':
            w = w.contiguous().view(s[0], s[1] * s[2] * s[3])
        if bias is not None:
            b = bias
            w = torch.cat([w, b.view(b.shape[0], 1)], dim=1)
        wn = torch.mm(torch.mm(ggt, w), xxt)
        wn *= state['num_locations']  # TODO check
        return (w * wn).sum()

    def _compute_covs(self, group, state, nbatches):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            x = F.conv2d(x, group['gathering_filter'],
                         stride=mod.stride, padding=mod.padding,
                         groups=mod.in_channels)
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        n = float(x.shape[1] * nbatches)
        if 'xxt' not in state:
            state['xxt'] = torch.mm(x, x.t()) / n
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(), beta=1., alpha=1. / n)
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        n = float(gy.shape[1] * nbatches)
        if 'ggt' not in state:
            state['ggt'] = torch.mm(gy, gy.t()) / n
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(), beta=1., alpha=1. / n)

    def _get_gathering_filter(self, mod):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh * j + kw * kh * i, 0, j, k] = 1
        return g_filter

    def get_pr(self):
        """Returns the PR of the eig. of F for the whole network."""
        sum_eig = 0
        sum_eig2 = 0
        for group in self.param_groups:
            weight = group['params'][0]
            state = self.state[weight]
            s, s2 = self._get_eigenvalues_one_module(state)
            sum_eig += s
            sum_eig2 += s2
        return (sum_eig ** 2) / sum_eig2

    def _get_eigenvalues_one_module(self, state):
        """Returns the sum and sum of squared eig. of F. of a module."""
        ex = torch.symeig(state['xxt'])[0]
        eg = torch.symeig(state['ggt'])[0]
        eigenvalues = torch.ger(ex, eg).view(-1)
        eigenvalues *= state['num_locations']  #TODO check
        s = eigenvalues.sum()
        s2 = (eigenvalues ** 2).sum()
        return s, s2


class ComputeFisherRaoNorm(object):
    def __init__(self):
        pass

    def __call__(self, results, name, set_name, analysis_loader, training_loader, model, optimizer,
                 device):

        batch_size = analysis_loader[0][0].size(0)
        nsamples = len(analysis_loader) * batch_size

        metric = FisherRaoNormKFAC(model)

        criterion = torch.nn.CrossEntropyLoss()

        for batch_idx, (data, target) in enumerate(analysis_loader):
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            loss = criterion(output, target)
            loss.backward()
            # Note there is no optimizer step here.
            with torch.no_grad():
                metric.update_stats(nsamples // batch_size)

        return {'kfac_fisher_rao_norm.empirical': metric.get_norm().item(),
                'pr_fisher.empirical': metric.get_pr().item()}
