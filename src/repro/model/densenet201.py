from repro.model.densenet import DenseNet, distribute


def build(input_size, num_classes, distributed=0):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32))

    return distribute(model, distributed)
