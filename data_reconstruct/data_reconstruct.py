import warnings

import torch

from . import backend
from . import model_classes


DEFAULT_JOINT_EMBED_KWARGS = {
    'project_mode': 'tsne',
    'epoch_pd': 2000,
    'log_pd': 200,
    'epoch_DNN': 100,
    'log_DNN': 50,
}
DEFAULT_NN_KWARGS = {
    'epochs': 100,
    'log_epoch': 50,
}


def run_pipeline(
    *datasets,
    train_idx=None,
    validation_idx=None,
    embedding_dim=10,
    hidden_dim=10,
    output_cols=[0],
    embedding_kwargs=DEFAULT_JOINT_EMBED_KWARGS,
    nn_kwargs=DEFAULT_NN_KWARGS,
):
    """
    Run the full suite.  The last modality is the one to fill in while the first
    should be the most reliable.

    ``train_idx`` is the # of training samples
    ``validation_idx`` is the optional # of validation samples.  By default,
        this will be the size of the last dataset
    It is assumed that the datasets are ordered training->validation->to_predict
    """
    if train_idx is None:
        train_idx = len(datasets[0])
        warnings.warn('``train_idx`` was not defined, using size of first dataset instead')
    if validation_idx is None and len(datasets[-1]) > train_idx:
        validation_idx = len(datasets[-1])
    if validation_idx is not None:
        maximal_idx = max(train_idx, validation_idx)
    else:
        maximal_idx = train_idx
    for i in range(len(datasets) - 1):
        assert len(datasets[i]) >= maximal_idx, \
            f'All datasets must have >= {maximal_idx} samples'

    # Perform embedding
    embedding_input = [*datasets]
    embedding_input[-1] = embedding_input[-1][:train_idx]

    joint_embedding = backend.joint_embed(
        *embedding_input,
        output_dim=embedding_dim,
        **embedding_kwargs,
    )

    # Predict last modality
    print('-' * 33)
    print('Mapping to last dataset...')
    model_train_X = joint_embedding[0][:train_idx]
    model_train_y = datasets[-1][:train_idx][:, output_cols]

    training_loader = backend.create_dataloader(model_train_X, model_train_y)
    model = model_classes.Model(embedding_dim, len(output_cols), hidden_dim=hidden_dim)
    backend.train_model(model, training_loader, **nn_kwargs)

    # Run validation
    if validation_idx is not None:
        model_validation_X = joint_embedding[0][train_idx:validation_idx]
        model_validation_y = datasets[-1][train_idx:validation_idx][:, output_cols]
        validation_loader = backend.create_dataloader(model_validation_X, model_validation_y)
        backend.run_validation(model, validation_loader)

    return model(torch.Tensor(joint_embedding[0])).detach().numpy()