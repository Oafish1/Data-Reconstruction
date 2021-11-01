# Data Reconstruct
This tool serves to fill in data gaps using auxiliary datasets.  This pipeline is based on the `UnionCom` method which is used outside its original scope of bioinformatics datasets.  `UnionCom` mappings are utilized to predict missing data from any number of datasets through the use of a thin neural network present after a joint embedding is generated.

## Usage
The pipeline takes as input any number of datasets in `np.array` form.  The pipeline can then be run with
```python
run_pipeline(*datasets, output_cols=output_cols)
```
`output_cols` defines the columns in the final dataset to be predicted.  The first modality should be the most reliable dataset, as it is the primary dataset used to fill in any data gaps.  The final modality should be the best-formatted dataset, and will be the dataset to be filled.

If you additionally wish to split into training/validation/testing sets, the arguments `train_idx` and `validation_idx` can be provided
```python
run_pipeline(..., train_idx=train_idx, validation_idx=validation_idx)
```
These should be the lengths of training and validation sets, respectively.  It is assumed that the data in each dataset is ordered by `training->validation->testing/unknown`.

Arguments can also be provided to `UnionCom` and the neural network directly using
```python
run_pipeline(..., embedding_kwargs=..., nn_kwargs=...)
```
However, please note that `output_dim` must be passed directly to `run_pipeline` as `embedding_dim`, rather than in `embedding_kwargs`.
