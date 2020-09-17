"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


def main():

    # fill NaN's with <UNK> token
    unk_token = '<UNK>'

    # retrieve dataset
    start = time.time()
    if os.path.exists('raw'):
        df = pd.read_csv('raw/churn.csv')
        del df['customerID']
    else:
        exit('directory raw does not exist!')
    print('time to load churn: {:.3f}s'.format(time.time() - start))

    # define columns
    label_col = 'Churn'
    feature_col = list(df.columns)
    feature_col.remove(label_col)

    # nan rows
    nan_rows = df[df.isnull().any(axis=1)]
    print('nan rows: {}'.format(len(nan_rows)))

    # fit encoders and fill in NaNs with unknown token or mean value
    encoders = {}
    for col in feature_col:
        if str(df[col].dtype) == 'object':
            df[col] = df[col].fillna(unk_token)
            encoders[col] = OrdinalEncoder().fit(df[col].to_numpy().reshape(-1, 1))
        else:
            df[col] = df[col].fillna(int(df[col].mean()))
    label_encoder = LabelEncoder().fit(df[label_col])

    # transform dataframe
    new_df = df.copy()
    for col in feature_col:
        if col in encoders:
            new_df[col] = encoders[col].transform(new_df[col].to_numpy().reshape(-1, 1))
    new_df[label_col] = label_encoder.transform(new_df[label_col])

    # show difference
    print('df')
    print(df.head(5))
    print(new_df.head(5))

    # save to numpy format
    print('saving to data.npy...')
    np.save('data.npy', new_df.to_numpy())
    np.save('feature.npy', feature_col)


if __name__ == '__main__':
    main()
