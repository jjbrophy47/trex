Amazon Dataset
---
* Download the following files from [https://www.kaggle.com/c/amazon-employee-access-challenge/data](https://www.kaggle.com/c/amazon-employee-access-challenge/data) to this directory.
    * `train.csv`.
    * `test.csv`.

* Preprocess the data.
    * Run `python3 preprocess.py` with arguments:
    	* `--processing`: `standard` or `categorical`. If `standard`, perform one-hot encoding on all categorical variables. Otherwise, if `categorical`, leave all categorical features as is.
    * Preprocessed data outputs to `[processing]/`.
