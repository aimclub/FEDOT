from torch import nn


class DummyOverComplicatedNeuralNetwork(nn.Module):
    def __init__(self,
                 input_dim=1000,
                 output_dim=10):
        super(DummyOverComplicatedNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(self.input_dim * self.input_dim, 1000)
        self.linear2 = nn.Linear(1000, 2000)
        self.linear3 = nn.Linear(2000, output_dim)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, self.input_dim * self.input_dim)  # .to(torch.float32)
        x = self.relu(self.linear1(x))  # .to(torch.float32)
        x = self.relu(self.linear2(x))  # .to(torch.float32)
        x = self.linear3(x)  # .to(torch.float32)
        return x
