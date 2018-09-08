from torch.optim import SGD


def build(model, lr, momentum, weight_decay):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
