"""
Simple integration tests to make sure the tree explainer works for all models, encodings, and binary / multi-class
datasets. These do NOT test correctness of the values returned by the tree explainer.
"""
from experiments.scripts.sample import example


def test_model(model='lgb', encodings=['tree_path', 'tree_output'],
               datasets=['iris', 'breast', 'wine', 'medifor']):

    for encoding in encodings:
        for dataset in datasets:
            print('\nTEST', model, encoding, dataset)
            example(model=model, encoding=encoding, dataset=dataset, timeit=True)


def main():
    test_model(model='rf')
    test_model(model='gbm')
    test_model(model='lgb')
    test_model(model='cb')
    test_model(model='xgb')


if __name__ == '__main__':
    main()
