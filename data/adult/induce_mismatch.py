"""
Induce domain mismatch between train and test.
Specifically, take the 395 people age <= 17 in the train data and
filter out 75% of them, and flip the labels for 85% of the remaining
samples.
"""
import os
import argparse
from datetime import datetime

import numpy as np

from preprocess import get_logger


def main(args):

    # create_logger
    assert os.path.exists(args.in_dir)
    logger = get_logger(os.path.join(args.in_dir, 'log_mismatch.txt'))
    logger.info('{}'.format(args))
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # target column
    age_col = 0

    # retrieve dataset
    train = np.load(os.path.join(args.in_dir, 'train.npy'), allow_pickle=True)
    test = np.load(os.path.join(args.in_dir, 'test.npy'), allow_pickle=True)

    indices = np.where(train[:, age_col] <= 17)[0]
    logger.info('\nTrain:')
    logger.info('no. instances in which age <= 17: {:,}'.format(len(indices)))
    logger.info('  -->no. pos. instances: {:,}'.format(np.sum(train[indices][:, -1])))

    # randomly sample a subset of people whose age <= 17
    np.random.seed(args.seed)
    subset_indices = np.random.choice(indices, size=int(len(indices) * args.subset_frac),
                                      replace=False)

    # of the above subset, randomly flip a percentage of their labels
    np.random.seed(args.seed)
    subsubset_indices = np.random.choice(subset_indices, size=int(len(subset_indices) * args.flip_frac),
                                         replace=False)

    remove_indices = np.setdiff1d(indices, subset_indices)
    train[subsubset_indices, -1] = 1
    train = np.delete(train, remove_indices, axis=0)

    indices = np.where(train[:, 0] <= 17)[0]
    logger.info('\nNew Train:')
    logger.info('no. instances in which age <= 17: {:,}'.format(len(indices)))
    logger.info('  -->no. pos. instances: {:,}'.format(np.sum(train[indices][:, -1])))

    test_indices = np.where(test[:, 0] <= 17)[0]
    logger.info('\nTest:')
    logger.info('no. instances in which age <= 17: {:,}'.format(len(test_indices)))
    logger.info('  -->no. pos. instances: {:,}'.format(np.sum(test[test_indices][:, -1])))

    # save to numpy format
    logger.info('\nsaving to train.npy...')
    np.save(os.path.join(args.in_dir, 'train_mismatch.npy'), train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='categorical', help='input data directory.')
    parser.add_argument('--subset_frac', type=float, default=0.25, help='subset size of age <= 17 samples.')
    parser.add_argument('--flip_frac', type=float, default=0.85, help='percentage of samples to flip labels.')
    parser.add_argument('--seed', type=int, default=1, help='random state.')
    args = parser.parse_args()
    main(args)
