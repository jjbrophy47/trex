Census Dataset
---
* Download the following files from [https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29) to this directory.
	* `census-income.data`.
	* `census-income.test`.

* Preprocess the data.
    * Run `python3 preprocess.py` with arguments:
    	* `--processing`: `standard` or `categorical`. If `standard`, perform one-hot encoding on all categorical variables. Otherwise, if `categorical`, leave all categorical features as is.
    * Preprocessed data outputs to `[processing]/`.
