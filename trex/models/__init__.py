"""
Binary Kernel Logistic regression and SVM using a modified version
of SKlearn's (version 0.24.1) adapted version of Liblinear.

Also includes a wrapper for KNN to use tree-extracted features.

Reference:
https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/svm
"""
from ._classes import SVM
from ._classes import KLR
from ._classes import KNN
