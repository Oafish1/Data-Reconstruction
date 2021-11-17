import matplotlib.pyplot as plt
import numpy as np
import torch

from .backend import create_dataloader, train_model
from .defaults import DEFAULT_NN_KWARGS
from .model_classes import Model


def generate_data(
    mod1_shape=(100, 5),
    relationship_complexity=3,
    training_pct=.8,
):
    """Generate fake data for testing usage"""
    relationship_shape = (mod1_shape[1], relationship_complexity)
    mod2_shape = (mod1_shape[0], relationship_shape[1])

    # Generate data
    mod1 = np.random.rand(*mod1_shape)
    relationship = np.random.rand(*relationship_shape)
    mod2 = np.dot(mod1, relationship)

    # Split data
    split_idx = int(training_pct * mod2_shape[0])
    mod2_training = mod2[:split_idx]
    mod2_validation = mod2[split_idx:]

    return mod1, relationship, mod2, mod2_training, mod2_validation, split_idx


def plot_example_embedding(
    joint_embedding,
    split_idx,
    dim_to_plot=[0, 1],
    axis_bounds=None,
):
    """Plot select dimensions of a joint embedding, includes 'missing' data"""
    plt.subplot(1, 2, 1)
    plt.title('First Modality')
    plt.scatter(
        *np.transpose(joint_embedding[0][:split_idx, dim_to_plot]),
        color='blue',
        label='Present Points',
    )
    plt.scatter(
        *np.transpose(joint_embedding[0][split_idx:, dim_to_plot]),
        color='red',
        label='Missing Points',
    )
    plt.legend()
    if axis_bounds is not None:
        plt.axis(axis_bounds)

    plt.subplot(1, 2, 2)
    plt.title('Last Modality')
    plt.scatter(*np.transpose(joint_embedding[-1][:, :2]), color='orange')
    if axis_bounds is not None:
        plt.axis(axis_bounds)


def plot_example_results(labels, logits, split_idx=None):
    """Plot the results of the mapping"""
    fig, ax = plt.subplots()
    plt.title('Prediction vs Reality')
    plt.xlabel('Truth')
    plt.ylabel('Prediction')
    if split_idx is None:
        ax.scatter(labels, logits, color='green')
    else:
        ax.scatter(labels[:split_idx], logits[:split_idx], color='blue')
        ax.scatter(labels[split_idx:], logits[split_idx:], color='red')

    # Make a line y=x
    # https://stackoverflow.com/questions/25497402/adding-y-x-to-a-matplotlib-scatter-plot-if-i-havent-kept-track-of-all-the-data
    ax_min = np.min([ax.get_xlim(), ax.get_ylim()])
    ax_max = np.max([ax.get_xlim(), ax.get_ylim()])
    lims = [ax_min, ax_max]

    ax.plot(lims, lims, '-', alpha=0.5, color='green')
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def predict_from_data(
    source,
    target,
    split_idx=None,
    nn_kwargs=DEFAULT_NN_KWARGS,
):
    return_split_idx = False
    if split_idx is None:
        split_idx = int(.8 * target.shape[0])
        return_split_idx = True

    training_loader = create_dataloader(source[:split_idx], target[:split_idx])
    model = Model(source.shape[1], target.shape[1], hidden_dim=20)
    train_model(model, training_loader, **nn_kwargs)
    ret = (model(torch.Tensor(source)).detach().numpy(),)

    if return_split_idx:
        ret += (split_idx,)
    return _return_tuple(ret)


def _return_tuple(tup):
    return tup[0] if len(tup) == 1 else tup
