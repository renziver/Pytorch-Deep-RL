import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward_pass(self, x):
        layer1_out = self.layer1(x)
        activation1 = self.relu(layer1_out)
        output = self.layer2(activation1)
        return output