import torch
import numpy as np


def reinforce(observation, labels, reward, baseline):
    cross_entropy = torch.sum(labels*torch.log(observation), dim=1)
    adv = reward - baseline
    loss = (-cross_entropy * adv).mean()
    return loss


def get_reward(predicted_label, true_label):
    correct = torch.tensor([np.where(r == 1)[0][0] for r in true_label]).numpy()
    p_values, p_indices = torch.max(predicted_label, 1)
    predicted = p_indices.numpy()
    reward = np.sum(correct == predicted)
    return reward
