from torch.optim import Adam


def build(model, lr, betas, weight_decay):
    return Adam(model.parameters(), lr=lr, betas=list(betas), weight_decay=weight_decay)
