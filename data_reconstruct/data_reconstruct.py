import warnings

import numpy as np
import torch

from . import backend
from . import model_classes
from .defaults import DEFAULT_JOINT_EMBED_KWARGS, DEFAULT_NN_KWARGS


def anonymize(
    *datasets,
    embedding_dim=10,
    embedding_kwargs=DEFAULT_JOINT_EMBED_KWARGS,
):
    """
    Run the suite until embedding, functioning as anonymization without
    specifically identifiable records.  Can be run on any number of datasets.
    This is functionally equivalent to UnionCom.

    Returns
    -------
    Tuple of embeddings generated from each dataset.  If only one embedding is
    to be sent, it should be generated from the most reliable dataset.
    """
    joint_embedding = backend.joint_embed(
        *datasets,
        output_dim=embedding_dim,
        **embedding_kwargs,
    )

    return joint_embedding


def run_pipeline(
    *datasets,
    column_fill=False,
    train_idx=None,
    validation_idx=None,
    embedding_dim=10,
    hidden_dim=20,
    output_cols=[0],
    std_function_and_inv=(
        lambda x: x,
        lambda x: x,
    ),
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

    ``column_fill``, if True, does not use existing data from column in joint
        embedding.  Set to False if you are trying to predict entirely missing
        samples
    Ex.
        column_fill = True
        a b c  1 x 3
        a b c  1 x 3
        a b c  1 x 3
        column_fill = False
        a b c  1 2 3
        a b c    x
        a b c    x
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
    if column_fill:
        masked_output_cols = np.ma.array(
            np.arange(embedding_input[-1].shape[1]),
            mask=False,
        )
        for i in output_cols:
            masked_output_cols.mask[i] = True

        # embedding_input[-1] = embedding_input[-1][
        #     :,
        #     masked_output_cols.compressed(),
        # ]

    else:
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
    model_train_y = std_function_and_inv[0](model_train_y)

    training_loader = backend.create_dataloader(model_train_X, model_train_y)
    model = model_classes.Model(embedding_dim, len(output_cols), hidden_dim=hidden_dim)
    model.train()
    backend.train_model(model, training_loader, **nn_kwargs)

    # Run validation
    model.eval()
    if validation_idx is not None:
        model_validation_X = joint_embedding[0][train_idx:validation_idx]
        model_validation_y = datasets[-1][train_idx:validation_idx][:, output_cols]
        validation_loader = backend.create_dataloader(model_validation_X, model_validation_y)
        backend.run_validation(model, validation_loader)

    return std_function_and_inv[1](
        model(torch.Tensor(joint_embedding[0])).detach().numpy()
    )
