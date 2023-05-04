from sklearn.datasets import load_svmlight_file, load_digits
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp
from lightning.classification import AdaGradClassifier, SVRGClassifier, SGDClassifier, SAGAClassifier, SAGClassifier
from functools import partial

from sklearn.model_selection import train_test_split

def load_data(dataset, train = False):
    if dataset in ['a9a', 'w8a']:
        X_train, y_train = load_svmlight_file('../datasets/{dataset}/{dataset}'.format(dataset = dataset))
        X_test, y_test = load_svmlight_file('../datasets/{dataset}/{dataset}.t'.format(dataset = dataset))
        
        if dataset == 'a9a' : X_test = np.append(np.array(X_test.todense()), np.zeros((X_test.shape[0], 1)), axis=1)
        
        return X_train, y_train, X_test, y_test

    if dataset == 'mnist':
        X, y = load_digits(return_X_y=True)
        
        # train test split sklearn
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

        return X_train, y_train, X_test, y_test


def train_test_model(dataset, optimizer, X_train, y_train, X_test, y_test, eta):
    clf = SGDClassifier(loss = 'log', penalty= 'l2',  eta0 = eta, max_iter = 100, random_state = 0, alpha = 1e-2)
    if optimizer == 'sgd':
        clf = SGDClassifier(loss = 'log', penalty= 'l2',  eta0 = eta, max_iter = 100, random_state = 0, alpha = 1e-2)
    if optimizer == 'adagrad':
        clf = AdaGradClassifier(loss = 'log', l1_ratio = 0,  eta = eta, n_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-25)
    if optimizer == 'svrg':
        clf = SVRGClassifier(loss = 'log', eta = eta, max_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-24)
    if optimizer == 'saga':
        clf = SAGAClassifier(loss = 'log', beta = 0, eta = eta, max_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-24)
    if optimizer == 'sag':
        clf = SAGClassifier(loss = 'log', beta = 0, eta = eta, max_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-24)
    
    clf.fit(X_train, y_train)
    return 1 - clf.score(X_test, y_test)


def run_hyperopt(dataset):
    hyper_params = {}
    for optimizer in ['sgd', 'svrg', 'saga', 'adagrad', 'sag']:

        X_train, y_train, X_test, y_test = load_data(dataset)
        # X_test, y_test = load_data(dataset, train = False)

        space = hp.uniform('eta', 1e-7, 1e-1)

        best = fmin(fn = partial(train_test_model, optimizer, dataset, X_train, y_train, X_test, y_test), space = space,
                    max_evals = 100) 

        # print
        print(dataset, optimizer, best)
        hyper_params[optimizer] = best
    return hyper_params

if __name__ == '__main__':

    
    for dataset in ['a9a', 'w8a', 'mnist']:
        for optimizer in ['sgd', 'svrg', 'saga', 'adagrad', 'sag']:

            X_train, y_train, X_test, y_test = load_data(dataset)
            # X_test, y_test = load_data(dataset, train = False)

            space = hp.uniform('eta', 1e-7, 1e-1)

            best = fmin(fn = partial(train_test_model, optimizer, dataset, X_train, y_train, X_test, y_test), space = space,
                        max_evals = 100) 

            # print
            print(dataset, optimizer, best)

