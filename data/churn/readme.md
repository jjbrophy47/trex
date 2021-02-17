Churn Dataset
---
* Download the following files from [https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn) to this directory.
    * `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

* Preprocess the data.
    * Run `python3 preprocess.py` with arguments:
    	* `--processing`: `standard` or `categorical`. If `standard`, perform one-hot encoding on all categorical variables. Otherwise, if `categorical`, leave all categorical features as is.
    * Preprocessed data outputs to [processing]/`.
