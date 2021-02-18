Adult Dataset
---
* Download the following files from [https://archive.ics.uci.edu/ml/machine-learning-databases/adult/](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/) to this directory.
	* `adult.data`.
	* `adult.test`.

* Preprocess the data.
    * Run `python3 preprocess.py` with arguments:
    	* `--processing`: `standard` or `categorical`. If `standard`, perform one-hot encoding on all categorical variables. Otherwise, if `categorical`, leave all categorical features as is.
    * Preprocessed data outputs to `[processing]/`.

* Induce domain mistmatch.
	* Run `python3 induce_mismatch.py` with arguments:
		* `--in_dir`: Input data directory, `standard` or `categorical`.
	* This creates a `[processing]/train_mismatch.npy`.