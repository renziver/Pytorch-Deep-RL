import torch
import numpy as np


def reinforce(observation, labels, reward, baseline):
    """Return the value of the loss based on REINFORCE algorithm

    Args:
        observation -- the observation (weights) of the model.
        labels -- the target classes of the dataset.
        reward -- the returns from each episode.
        baseline -- the average value of returns.
    """
    cross_entropy = torch.sum(labels*torch.log(observation), dim=1)
    adv = reward - baseline
    loss = (-cross_entropy * adv).mean()
    return loss


def get_reward(predicted_label, true_label):
    """Return the scalar value of the earned reward from an episode (number of correct predictions)

    Args:
        predicted_label -- the prediction after following a policy.
        true_label -- the ground truth/target label.
    """
    correct = torch.tensor([np.where(r == 1)[0][0] for r in true_label]).numpy()
    p_values, p_indices = torch.max(predicted_label, 1)
    predicted = p_indices.numpy()
    reward = np.sum(correct == predicted)
    return reward
