# Data Reconstruct
This tool serves to fill in data gaps using auxiliary datasets.  This pipeline is heavily based on the `UnionCom` method which is used outside its original scope of bioinformatics datasets, which is the main contribution of this library.  `UnionCom` mappings are utilized to predict missing data from any number of datasets through the use of a thin neural network present after a joint embedding is generated.

This particular methodology is utilized in the hope that `UnionCom`'s calculation of the `F` matrix approximation (emulation of one modality through the formula `FKF^T`) imbues information about the construction of one modality from another.  After mapping to a common space, this information would hopefully remain intact, enough so that we can perform reconstruction -- even where there wasn't data to begin with.

## Blank Data Reconstruction
### Usage
The main purpose of `run_pipeline` is to provide functionality for filling in data gaps using auxiliary datasets.  This is achieved through the use of `UnionCom` and a thin neural network.

### Process
The pipeline takes as input any number of datasets in `np.array` form.  The datasets do not need to be aligned, but must be ordered as noted below.  The pipeline can then be run with
```python
run_pipeline(*datasets, output_cols=output_cols)
```
Output is a prediction of the feature(s) specified in `output_cols` present in the final dataset from `datasets`.

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

### Limitations
It is assumed that the data has the same 'shape'.  In other words, the datasets are generally related in terms of variance.  This is necessary so that an effective joint embedding can be produced.

## Anonymization
### Usage
This function is generally meant to anonymize data from any number of datasets.  The `UnionCom` algorithm is used **alone** and is the sole codebase for this particular application.

### Process
`anonymize` takes similar arguments to `run_pipeline`
```python
anonymize(*datasets, embedding_dim=embedding_dim)
```
Output is the mapping from `UnionCom`.  This will be a tuple of length `len(datasets)`.  Each entry is a mapping derived from each dataset, respectively.  In practice, it is likely and recommended that only one of these will be chosen.  If this is the case, the mapping chosen should correspond to the most reliable/relevant dataset.

`embedding_dim` represents the number of features to use when generating the embedding.  By default, this is set to `10`.  In general, higher `embedding_dim` will lead to greater resolution (and, therefore, more predictive power) but more concerns for reverse-engineering.

Tweaking the embedding parameters can be done through `embedding_kwargs`
```python
anonymize(..., embedding_kwargs=...)
```

### Limitations
If given sample data and anonymized data from the same pool, it is likely that data reconstruction could be performed similar to `run_pipeline` above.

`anonymize` has the same data constraints as `run_pipeline` above.

## Future Work
Currently, the `UnionCom` code used is from [caokai1073's repository](https://github.com/caokai1073/UnionCom).  When using 3+ datasets, the last dataset is made to be the 'definitive' one against which many comparisons are made.  A revision similar to that made in [ComManDo](https://github.com/Oafish1/ComManDo) may be performed to change this.  Additionally, `NLMA` mapping could be used for aligned datasets in anonymization.  However, this is purely speculative.
