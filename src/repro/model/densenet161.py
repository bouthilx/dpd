from repro.model.densenet import DenseNet, distribute


def build(input_size, num_classes, distributed=0):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))

    return distribute(model, distributed)
