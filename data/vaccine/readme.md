Vaccine Dataset
---
* Download the following files from [https://www.drivendata.org/competitions/66/flu-shot-learning/data/](https://www.drivendata.org/competitions/66/flu-shot-learning/data/) (you may have to register with DrivenData to access the dataset) to this directory.
	* `training_set_features.csv`.
	* `training_set_labels.csv`.

* Preprocess the data.
    * Run `python3 preprocess.py` with arguments:
    	* `--processing`: `standard` or `categorical`. If `standard`, perform one-hot encoding on all categorical variables. Otherwise, if `categorical`, leave all categorical features as is.
    * Preprocessed data outputs to `[processing]/`.
