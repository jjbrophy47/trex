"""
This script plots the average impact of the training instances (patients aged 40-50 and readmitted)
on each test instance (patients aged 40-50). This is plotted against the results from LeafInfluence.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def main():

    # get impact of each training instance on each test instance
    res_ours = np.load('ours/age_readmit.npy').T
    res_inf = np.load('influence/age_readmit.npy')

    # separate test instances into readmitted and non-readmitted patients
    test_age_ndx = np.load('test_age_ndx.npy')
    test_age_readmit_ndx = np.load('test_age_readmit_ndx.npy')
    _, test_age_readmit_context_ndx, _ = np.intersect1d(test_age_ndx, test_age_readmit_ndx, return_indices=True)
    test_age_noreadmit_context_ndx = np.setdiff1d(np.arange(len(test_age_ndx)), test_age_readmit_context_ndx)

    # compute the average impact of the training instances on each test instances
    ours_test_mean = np.mean(res_ours, axis=1)
    inf_test_mean = np.mean(res_inf, axis=1)

    # plot the results
    pearson = pearsonr(ours_test_mean, inf_test_mean)[0]

    fig, ax = plt.subplots()
    ax.scatter(ours_test_mean[test_age_noreadmit_context_ndx], inf_test_mean[test_age_noreadmit_context_ndx],
               color='purple', label='age=(40,50], y=0', marker='o', facecolors='none')
    ax.scatter(ours_test_mean[test_age_readmit_context_ndx], inf_test_mean[test_age_readmit_context_ndx],
               color='green', label='age=(40,50], y=1', marker='x')

    # ax.scatter(res_ours.flatten(), res_inf.flatten(),
    #            color='orange', label='age=(40,50]', marker='.', facecolors='none')

    ax.set_xlabel('ours (avg impact)')
    ax.set_ylabel('leafinfluence (avg influence)')
    ax.legend()
    plt.savefig('mismatch_modified.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
