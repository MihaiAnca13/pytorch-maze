import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 100
EPOCHS = 3


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 4)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")

        self.to(self.device)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.float()
        x = x.to(self.device)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        # print(x[0].shape[0] * x[0].shape[1] * x[0].shape[2])

        x = x.view(-1, 256) # 30976
        x = self.fc1(x)
        x = self.fc2(x)

        x = F.softmax(x, dim=1)

        return x

    def batch_train(self, train_X, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS):
        for epoch in range(epochs):
            for i in range(0, len(train_X), batch_size):
                batch_X = train_X[i:i + BATCH_SIZE]
                batch_y = train_y[i:i + BATCH_SIZE]

                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.zero_grad()
                outputs = self(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            # print(loss)


if __name__ == "__main__":
    from maze import Maze

    n = TorchModel()
    m = Maze(10, 10)
    a, _, _ = m.step(0, 1, show=False,)
    print(a.shape)
    a = torch.Tensor(a).view(-1, 1, m.MAX_X, m.MAX_Y)
    with torch.no_grad():
        b = n(a)
    print(b)
