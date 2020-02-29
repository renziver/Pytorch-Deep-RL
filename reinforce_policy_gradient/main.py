import matplotlib.pyplot as plt
import numpy as np
import torch
from reinforce_policy_gradient.policy import Policy
from reinforce_policy_gradient.reinforce import reinforce, get_reward
from reinforce_policy_gradient.utils import config_args, load_dataset, one_hot_embedding

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

input_size, hidden_size, num_classes, num_epochs, batch_size, \
    learning_rate, save_model = config_args('config.yaml', 'hyperparams')

training_set, test_set = load_dataset(batch_size)

model = Policy(input_size, hidden_size, num_classes).to(device)
loss_function = reinforce

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(training_set)
tau = []
loss_values = []
for epoch in range(num_epochs):
    for index, (images, labels) in enumerate(training_set):
        images = images.reshape(-1, input_size).to(device)
        labels = one_hot_embedding(labels.to(device), 10)

        outputs = model.forward_pass(images)
        reward = get_reward(outputs, labels)
        tau.append(reward)
        baseline = np.mean(tau)
        loss = loss_function(outputs, labels, reward, baseline)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index + 1) % batch_size == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Reward:{:.4f}'
                  .format(epoch + 1, num_epochs, index + 1, total_step, loss.item(), reward))
            loss_values.append(loss.item())

# Visualize training results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('REINFORCE algorithm training results')
ax1.plot(loss_values)
ax1.set_title('Loss over number of episodes')
ax2.plot(tau)
ax2.set_title('Rewards over number of episodes')
fig.savefig('assets/images/results.png')

# Test model on test data
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_set:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model.forward_pass(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 1000 test images: {} %'
          .format(100 * correct / total))

if save_model:
    torch.save(model.state_dict(), 'assets/models/model.ckpt')
