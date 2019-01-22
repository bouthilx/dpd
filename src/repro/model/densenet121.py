from repro.model.densenet import DenseNet, distribute


def build(input_size, num_classes, distributed=0):
    model = DenseNet(input_size=input_size, num_init_features=64, growth_rate=32,
                     block_config=(6, 12, 24, 16))

    return distribute(model, distributed)
