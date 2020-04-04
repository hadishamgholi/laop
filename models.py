from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv_layer = nn.Sequential(
        #     nn.Conv2d(3, 6, 5),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(),


        # )
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(64, 80, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(80, 96, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(96, 112, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(112, 128, kernel_size=3),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x