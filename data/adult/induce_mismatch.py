"""
Induce domain mismatch between train and test.
Specifically, take the 395 people age <= 17 in the train data and
filter out 75% of them, and flip the labels for 85% of the remaining
samples.
"""
import numpy as np


def main(seed=1, subset_frac=0.25, flip_frac=0.85):

    # retrieve dataset
    train = np.load('train.npy')
    test = np.load('test.npy')

    indices = np.where(train[:, 0] <= 17)[0]
    print('Train:')
    print('num instances where age <= 17: {}'.format(len(indices)))
    print('  -->pos instances: {}'.format(np.sum(train[indices][:, -1])))

    np.random.seed(seed)
    subset_indices = np.random.choice(indices, size=int(len(indices) * subset_frac),
                                      replace=False)
    np.random.seed(seed)
    subsubset_indices = np.random.choice(subset_indices, size=int(len(subset_indices) * flip_frac),
                                         replace=False)

    remove_indices = np.setdiff1d(indices, subset_indices)
    train[subsubset_indices, -1] = 1
    train = np.delete(train, remove_indices, axis=0)

    indices = np.where(train[:, 0] <= 17)[0]
    print('New Train:')
    print('num instances where age <= 17: {}'.format(len(indices)))
    print('  -->pos instances: {}'.format(np.sum(train[indices][:, -1])))

    test_indices = np.where(test[:, 0] <= 17)[0]
    print('Test:')
    print('num instances where age <= 17: {}'.format(len(test_indices)))
    print('  -->pos instances: {}'.format(np.sum(test[test_indices][:, -1])))

    # save to numpy format
    print('saving to train.npy...')
    np.save('train_mismatch.npy', train)


if __name__ == '__main__':
    main()
