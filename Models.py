
import torch.nn as nn

class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=0)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=0)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=0)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=0)
        self.MaxPool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(73728, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.MaxPool1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.MaxPool2(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.MaxPool3(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x=self.sigmoid(self.fc3(x))

        return x

