#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import sqrt
from numpy.random import multivariate_normal, randn
import math
from sklearn.utils import shuffle
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=2)


# # Parameters
#server
n_features = 2000
sparsity = 20
sigma = 1
N = 1000 # size of dataset
V = 40 # number of blocks for median
K0 = math.ceil(math.log(V/3,2)+2)
# Kmax = max(3, math.ceil(math.log(V/3,2)+1))
Kmax = 4
grid_lamda = np.exp(np.arange(-2,4,.5))
outliers_plus_heavytail_range = np.arange(0,151,4)
E = 100 # number of experiments (for averaging)

#Joons
# n_features = 30
# sparsity = 10
# sigma = 1
# N = 1000 # size of dataset
# V = 40 # number of blocks for median
# K0 = math.ceil(math.log(V/3,2)+2)
# # Kmax = max(3, math.ceil(math.log(V/3,2)+1))
# Kmax = 4
# grid_lamda = np.exp(np.arange(-2,4,.5))
# outliers_plus_heavytail_range = np.arange(0,151,4)
# E = 2 # number of experiments (for averaging)
# CALCUL EN PARALELLE SUR 12 COEURS
#parallel = 12

#Guillaume
# n_features = 20
# sparsity = 5
# sigma = 1
# N = 100 # size of dataset
# V = 10 # number of blocks for median
# K0 = math.ceil(math.log(V/3,2)+2)
# # Kmax = max(3, math.ceil(math.log(V/3,2)+1))
# Kmax = 3
# grid_lamda = np.exp(np.arange(-2,4,1))
# outliers_plus_heavytail_range = np.arange(0,100,10)
# E = 1 # number of experiments (for averaging)



# # Preliminary computations
subsamples = {}
for K in range(3,math.floor(math.log(N, 2))+1):
    for k in range(1,2**K+1):
        subsamples[(K,k)] = np.arange(math.floor((k-1)*N/2**K),math.floor(k*N/2**K))


mom_partition = set()
K = math.ceil(math.log(V/3,2)+2)
for k in range(1,2**K+1):
    mom_partition.add((K,k))



def beta_func(n_features, sparsity):
    idx = np.arange(n_features)
    beta = (n_features/10)*(-1) ** (abs(idx - 1)) * np.exp(-idx / 10.)
    sel = np.random.permutation(n_features)
    sel1 = sel[0:int(sparsity/4)]
    beta[sel1] = 10
    sel11 = sel[int(sparsity/4):int(sparsity/2)]
    beta[sel11] = -10
    sel0 = sel[sparsity:]
    beta[sel0] = 0.
    return beta



def data1(n_samples, beta, sigma):
    n_features = beta.size
    cov = np.identity(n_features)
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    Y = X.dot(beta) + sigma * randn(n_samples)
    return Y, X

def data2(n_outliers, n_features, type_outliers = 1, beta = 1, rho=1):
    if type_outliers == 1:
        Y = np.ones(n_outliers)
        X = np.ones((n_outliers, n_features))
    elif type_outliers == 2:
        Y = 10000*np.ones(n_outliers)
        X = np.ones((n_outliers, n_features))
    elif type_outliers == 3:
        Y = np.random.randint(2, size = n_outliers)
        X = np.random.rand(n_outliers, n_features)
    else:
        cov = np.identity(n_features)
        X = feature_mat(n_features, n_outliers, rho)
        Y = X.dot(beta) + sigma * randn(n_samples)
    return Y, X

def data3(n_heavy_tail, beta, deg = 2):
    n_features = beta.size
    cov = np.identity(n_features)
    X = multivariate_normal(np.zeros(n_features), cov, size=n_heavy_tail)
    Y = X.dot(beta) + np.random.standard_t(deg, size=n_heavy_tail)
    return Y, X



def data_merge_with_outlier_info(Y1, X1, Y2, X2, Y3, X3):
    Y = np.concatenate((Y1, Y2, Y3), axis=0)
    X = np.concatenate((X1, X2, X3), axis=0)
    outlier1p = np.concatenate((np.full(Y1.size,False), np.full(Y2.size,True), np.full(Y3.size, False)))
    outlier2p = np.concatenate((np.full(Y1.size,False), np.full(Y2.size,False), np.full(Y3.size, True)))
    return shuffle(Y, X, outlier1p, outlier2p)


def mom_generate_data(n_total, n_outliers, n_heavy_tail):
    n_samples = n_total - n_outliers - n_heavy_tail
    beta_0 = beta_func(n_features, sparsity)
    y1, X1 = data1(n_samples, beta_0,  sigma)
    y2, X2 = data2(n_outliers, n_features, type_outliers = 2, beta = 1, rho=1)
    y3, X3 = data3(n_heavy_tail, beta_0, deg = 2)
    y, X, outlier1p, outlier2p = data_merge_with_outlier_info(y1, X1, y2, X2, y3, X3)

    beta_0 = np.matrix(beta_0).T
    y = np.matrix(y).T
    X = np.matrix(X)
    return beta_0, y, X, outlier1p, outlier2p


# # MOM Selection functions


def mom_number_of_hyperparameters_m(N, grid_lamda, Kmax):
    return grid_lamda.size * 8 * (2 ** (Kmax - 2) - 1)


def mom_decompose_hyperparameter_m(ind_m, N, grid_lamda):
    # lamda varie d'abord: cinq blocs B consécutifs sont égaux
    # k varie de 1 à 2^K
    # K varie de 3 à Kmax
    # indice du lamda correspondant
    ind_lamda = ind_m % grid_lamda.size
    # "indice" du block correspondant
    ind_wo_lamda = math.floor(ind_m / grid_lamda.size)
    K = math.floor(math.log(2**3+ind_wo_lamda, 2))
    k = ind_wo_lamda - 8*((2**(K-3)-1))+1
    
    return ind_lamda, K, k, subsamples[(K,k)]


def mom_intersecting_blocks(K,k,K0):
    if K0 <= K:
        return {(K0,1+math.floor((k-1)/2**(K-K0)))}
    else:
        return {(K0, k0) for k0 in range(2**(K0-K)*(k-1), 2**(K0-K)*k)}

    mom_intersecting_blocks(1,1,K0)


def mom_estimator_selection(X, y, beta_0, V, grid_lamda, Kmax, outlier1p, outlier2p):
    global K0, mom_partition
    
    N = y.size
    M = mom_number_of_hyperparameters_m(N, grid_lamda, Kmax)
    if V > (N/8):
        print('V larger than N/8. Correcting')
        V = N/8
        # un truc où on peut append:
        # blocks = []
    estimators = []
    estimators_errors = np.zeros(M)
    empirical_errors_on_blocks = {}
    total_empirical_errors = np.zeros(M)
    nb_outliers_in_subsamples = [] 
    outlierp = list(np.logical_or(outlier1p,outlier2p))

    print('Computing estimators on subsamples')
    for ind_m in np.arange(0, M):
        # print('computing estimator for ind_m=', ind_m, '... ')
        ind_lamda, K, k, data_ind = mom_decompose_hyperparameter_m(ind_m, N, grid_lamda)
        nb_outliers_in_subsamples.append(sum(outlierp[data] for data in data_ind))
        # _, _, estimator = MOM_LASSO.MOM_ADMM(X[data_ind], y[data_ind], beta_0, 1, 100, grid_lamda[ind_lamda])
        lasso = linear_model.Lasso(alpha=grid_lamda[ind_lamda], fit_intercept=False)
        lasso.fit(X[data_ind], y[data_ind])
        estimator = np.transpose([lasso.coef_])

        estimators.append(estimator)
        estimators_errors[ind_m] = np.linalg.norm(beta_0-estimator)
        # compute empirical error on each test block
        for T in mom_partition:
            subsample = subsamples[T]
            empirical_errors_on_blocks[(ind_m,T)] = np.linalg.norm(X[subsample]*estimator-y[subsample])
            #best estimator?
    best_estimator_ind = np.argmin(estimators_errors)
    best_estimator = estimators[best_estimator_ind]
    
    print('Computing MOM-selection')
    max_over_m_prime = np.zeros(M)

    for ind_m in np.arange(0,M):
        medians = np.zeros(M)
        _, K, k, _ = mom_decompose_hyperparameter_m(ind_m, N, grid_lamda)
        # print('comparing estimators for ind_m=', ind_m)
        for ind_m_prime in np.arange(0,M):
            _, K_prime, k_prime, _ = mom_decompose_hyperparameter_m(ind_m_prime, N, grid_lamda)
            # comparison partition
            comparison_partition = mom_partition.copy()
            comparison_partition = comparison_partition - mom_intersecting_blocks(K,k,K0)
            comparison_partition = comparison_partition - mom_intersecting_blocks(K_prime,k_prime,K0)
            if len(comparison_partition) < V:
                raise ValueError('Problème: la parition de test est trop petite')
            while len(comparison_partition) > V:
                comparison_partition.pop()
                # la mediane sur V
            empirical_diffs = []
            
            for T in comparison_partition:
                empirical_diffs.append(empirical_errors_on_blocks[(ind_m,T)]-empirical_errors_on_blocks[(ind_m_prime,T)])
                # median over v
            medians[ind_m_prime] = np.median(empirical_diffs)
            # max over m_prime
        max_over_m_prime[ind_m] = max(medians)
        # argmin over m
    selected_m_ind = np.argmin(max_over_m_prime)
    selected_estimator = estimators[selected_m_ind]
    return estimators, estimators_errors, selected_m_ind, selected_estimator, best_estimator_ind, best_estimator, nb_outliers_in_subsamples



def compute_mom(nb_outliers_plus_heavytail):
    global subsamples, mom_partition
    beta_0, y, X, outlier1p, outlier2p = mom_generate_data(N, int(nb_outliers_plus_heavytail/2), int(nb_outliers_plus_heavytail/2))
    estimators, estimators_errors, selected_m_ind, selected_estimator, best_estimator_ind, best_estimator, nb_outliers_in_subsamples = mom_estimator_selection(X, y, beta_0, V, grid_lamda, Kmax, outlier1p, outlier2p)

    selected_estimator_error = np.linalg.norm(beta_0-selected_estimator)
    
    selected_ind_lamda, _, _, selected_data_ind = mom_decompose_hyperparameter_m(selected_m_ind, N, grid_lamda)
    best_estimator_ind_lamda, _, _, best_estimator_data_ind = mom_decompose_hyperparameter_m(best_estimator_ind, N, grid_lamda)

    nb_outliers2_in_selected = sum(outlier2p[data] for data in selected_data_ind)
    nb_outliers2_in_best = sum(outlier2p[data] for data in best_estimator_data_ind)
    
    nb_outliers1_in_selected = sum(outlier1p[data] for data in selected_data_ind)
    nb_outliers1_in_best = sum(outlier1p[data] for data in best_estimator_data_ind)
    
    nb_with_no_outlier = nb_outliers_in_subsamples.count(0)/len(grid_lamda)
    lowest_error_among_computed = min(estimators_errors)

    basic_estimators_errors = []
    for lamda in grid_lamda:
        # _, _, estimator = MOM_LASSO.MOM_ADMM(X, y, beta_0, 1, 100, lamda)
        lasso = linear_model.Lasso(alpha=lamda, fit_intercept=False)
        lasso.fit(X, y)
        estimator = np.transpose([lasso.coef_])

        basic_estimators_errors.append(np.linalg.norm(beta_0-estimator))
        lowest_error_among_basic = min(basic_estimators_errors)

    return selected_estimator_error, nb_outliers1_in_selected, nb_outliers1_in_best, nb_outliers2_in_selected, nb_outliers2_in_best, nb_with_no_outlier, lowest_error_among_computed, lowest_error_among_basic


# # Computations
#pool = multiprocessing.Pool(parallel)


selected_estimators_errors_all = []
nb_outliers1_in_selected_blocks_all = []
nb_outliers1_in_best_estimator_subsample_all = []
nb_outliers2_in_selected_blocks_all = []
nb_outliers2_in_best_estimator_subsample_all = []
nb_subsamples_with_no_outlier_all = []
lowest_error_among_computed_estimators_all = []
lowest_error_among_basic_estimators_all = []


for experiment in range(E):
	print('number of experiment:{}'.format(experiment))
	selected_estimators_errors = []
	nb_outliers1_in_selected_blocks = []
	nb_outliers1_in_best_estimator_subsample = []
	nb_outliers2_in_selected_blocks = []
	nb_outliers2_in_best_estimator_subsample = []
	nb_subsamples_with_no_outlier = []
	lowest_error_among_computed_estimators = []
	lowest_error_among_basic_estimators = []

	selected_estimators_errors, nb_outliers1_in_selected_blocks, nb_outliers1_in_best_estimator_subsample, nb_outliers2_in_selected_blocks, nb_outliers2_in_best_estimator_subsample, nb_subsamples_with_no_outlier, lowest_error_among_computed_estimators, lowest_error_among_basic_estimators = map(np.array, zip(*map(compute_mom, outliers_plus_heavytail_range)))

	selected_estimators_errors_all.append(selected_estimators_errors)
	nb_outliers1_in_selected_blocks_all.append(nb_outliers1_in_selected_blocks)
	nb_outliers1_in_best_estimator_subsample_all.append(nb_outliers1_in_best_estimator_subsample)
	nb_outliers2_in_selected_blocks_all.append(nb_outliers2_in_selected_blocks)
	nb_outliers2_in_best_estimator_subsample_all.append(nb_outliers2_in_best_estimator_subsample)
	nb_subsamples_with_no_outlier_all.append(nb_subsamples_with_no_outlier)
	lowest_error_among_computed_estimators_all.append(lowest_error_among_computed_estimators)
	lowest_error_among_basic_estimators_all.append(lowest_error_among_basic_estimators)


##Pickle dump
import pickle

filename = 'outliers_plus_heavytail_range.pickle'
data = outliers_plus_heavytail_range	
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

filename = 'selected_estimators_errors_all.pickle'
data = 	selected_estimators_errors_all
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

filename = 'nb_outliers1_in_selected_blocks_all.pickle'
data = nb_outliers1_in_selected_blocks_all	
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

filename = 'nb_outliers1_in_best_estimator_subsample_all.pickle'
data = 	nb_outliers1_in_best_estimator_subsample_all
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

filename = 'nb_outliers2_in_selected_blocks_all.pickle'
data = 	nb_outliers2_in_selected_blocks_all
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

filename = 'nb_outliers2_in_best_estimator_subsample_all.pickle'
data = 	nb_outliers2_in_best_estimator_subsample_all
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


filename = 'nb_subsamples_with_no_outlier_all.pickle'
data = nb_subsamples_with_no_outlier_all	
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

filename = 'lowest_error_among_computed_estimators_all.pickle'
data = 	lowest_error_among_computed_estimators_all
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

filename = 'lowest_error_among_basic_estimators_all.pickle'
data = 	lowest_error_among_basic_estimators_all
with open(filename, 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


# if __name__ == "__main__":
#     mom_kll()

