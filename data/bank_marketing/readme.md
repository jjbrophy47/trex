Bank Marketing Dataset
---
* Download the following files from [https://data.world/uci/bank-marketing](https://data.world/uci/bank-marketing) to this directory.
    * `bank-additional_bank-additional-full.csv`.

* Preprocess the data.
    * Run `python3 preprocess.py` with arguments:
    	* `--processing`: `standard` or `categorical`. If `standard`, perform one-hot encoding on all categorical variables. Otherwise, if `categorical`, leave all categorical features as is.
    * Preprocessed data outputs to `[processing]/`.
