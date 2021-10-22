# Data Reconstruct
This tool serves to fill in data gaps using auxiliary datasets.  This pipeline is based on the `UnionCom` method and has been tuned/modified for our purposes.

## Usage
The pipeline takes as input any number of datasets in `np.array` form.  The pipeline can then be run with
```python
run_pipeline(*datasets, output_cols=output_cols)
```
`output_cols` defines the columns in the final dataset to be predicted.  The first modality should be the most reliable dataset, as it is the primary dataset used to fill in any data gaps.  The final modality should be the best-formatted dataset, and will be the dataset to be filled.

If you additionally wish to split into training/validation/testing sets, the arguments `train_idx` and `validation_idx` can be provided
```python
run_pipeline(*datasets, train_idx=train_idx, validation_idx=validation_idx)
```
These should be the lengths of training and validation sets, respectively.  It is assumed that the data in each dataset is ordered by `training->validation->testing/unknown`.
