import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LeNet, self).__init__()
        n_channels = input_size[0]
        if n_channels == 1:
            self.conv1 = nn.Conv2d(n_channels, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1   = nn.Linear(50 * 4 * 4, 500)
            self.fc2   = nn.Linear(500, 84)
            self.fc3   = nn.Linear(84, num_classes)
        else:
            self.conv1 = nn.Conv2d(n_channels, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1   = nn.Linear(16*5*5, 120)
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, num_classes)

        self._initialize_weights()

    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = nn.functional.max_pool2d(out, 2)
        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


def build(input_size, num_classes):
    return LeNet(input_size=input_size, num_classes=num_classes)
