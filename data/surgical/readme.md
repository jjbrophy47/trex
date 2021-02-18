Surgical Dataset
---
* Download the following files from [https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification/version/1#](https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification/version/1#) to this directory.
	* `Surgical-deepnet.csv`.

* Preprocess the data.
    * Run `python3 preprocess.py` with arguments:
    	* `--processing`: `standard` or `categorical`. If `standard`, perform one-hot encoding on all categorical variables. Otherwise, if `categorical`, leave all categorical features as is.
    * Preprocessed data outputs to `[processing]/`.
