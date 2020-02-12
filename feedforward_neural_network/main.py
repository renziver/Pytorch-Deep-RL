import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from feedforward_neural_network.ffnn import NeuralNet
from feedforward_neural_network.utils import config_args, load_dataset

input_size, hidden_size, num_classes, num_epochs, batch_size, \
learning_rate, save_model = config_args('config.yaml', 'hyperparams')

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

training_set, test_set = load_dataset(batch_size)

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(training_set)
for epoch in range(num_epochs):
    for index, (images, labels) in enumerate(training_set):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model.forward_pass(images)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index + 1) % batch_size == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, index + 1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_set:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model.forward_pass(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 1000 test images: {} %'
         .format(100 * correct / total))

if save_model:
    torch.save(model.state_dict(),'assets/models/model.ckpt')