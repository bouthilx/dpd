from torch.optim import SGD


def build(model, lr, momentum):
    return SGD(model.parameters(), lr=lr, momentum=momentum)
