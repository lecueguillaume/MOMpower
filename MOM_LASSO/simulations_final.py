import numpy as np
from numpy.random import multivariate_normal, randn
from numpy.linalg import norm
from numpy import sqrt
from scipy.linalg import svd
from scipy.linalg.special_matrices import toeplitz
from scipy import linalg
import matplotlib.pylab as plt
import time
from sklearn.utils import shuffle
import pickle


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

def data4(n_outliers, n_features):
    Y = 10000*np.ones(n_outliers)
    X = np.ones((n_outliers, n_features))
    return Y, X

def data_merge(Y1, X1, Y2, X2):
    Y = np.concatenate((Y1, Y2), axis=0)
    X = np.concatenate((X1, X2), axis=0)
    return shuffle(Y, X)

def median_index(vector):
    med = np.median(vector)
    return np.nanargmin(np.abs(vector-med))

def block(X, y, x, x_prime, K):
    vect_means = []
    N = y.size
    for k in range(0,K):
        Xk, yk = X[k*N//K: (k+1)*N//K], y[k*N//K: (k+1)*N//K]
        excess_loss_k = linalg.norm(Xk.dot(x) - yk) ** 2 - linalg.norm(Xk.dot(x_prime) - yk) ** 2
        vect_means.append(excess_loss_k)
    idx_med = median_index(vect_means)
    #print(idx_med)
    return X[idx_med*N//K:(idx_med+1)*N//K], y[idx_med*N//K:(idx_med+1)*N//K]

def random_block(X, y, x, x_prime, K):
    vect_means = []
    N = y.size
    li = np.arange(N)
    np.random.shuffle(li)
    K = int(K)
    for k in range(0, K):
        ind = li[k*N//K: (k+1)*N//K]
        Xk, yk = X[ind], y[ind]
        excess_loss_k = linalg.norm(Xk.dot(x) - yk) ** 2 - linalg.norm(Xk.dot(x_prime) - yk) ** 2
        vect_means.append(excess_loss_k)
    idx_med = median_index(vect_means)
    ind = li[idx_med*N//K: (idx_med+1)*N//K]
    #print(idx_med)
    return X[ind], y[ind], ind

def obj(X, y, w, lamda):
    r = X*w-y;
    return np.sum(np.multiply(r,r))/2 +  lamda * np.sum(np.abs(w))

def subgrad(X, y, w, lamda):
    return  X.T*(X*w-y) + lamda*np.sign(w) 

def f_grad(X, y, w):
    return  X.T*(X*w-y) 

def soft_threshod(w, mu):
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))  

def smooth_grad(X, y, w, mu, lamda):
    temp = np.multiply((np.abs(w)<=mu),w/mu) + np.multiply((np.abs(w)>mu),np.sign(w)) 
    return X.T*(X*w-y) + lamda * temp

def mom_obj(X, y, x, x_prime, lamda):
    r = X*x - y
    r_prime = X*x_prime - y
    return np.sum(np.multiply(r,r))/2 +  lamda * np.sum(np.abs(x)) - np.sum(np.multiply(r_prime,r_prime))/2 +  lamda * np.sum(np.abs(x_prime))

## ADMM
def ADMM(X, y, beta_0, max_iter, lamda):
    dim = X.shape[1]
    w = np.matrix([0.0]*dim).T
    z = w
    u = w
    obj_ADMM, error_ADMM = [], []
    rho = 5
    print("Lauching ADMM solver...")
    for t in range(0, max_iter):
        #if (t%100==0):
        #    print('iter= {}, estimation error ADMM = {:3f}'.format(t, norm(beta_0 - w)))
        w = np.linalg.solve((X.T)*X + rho*np.identity(dim), X.T*y + rho*z - u)
        z = soft_threshod(w + u/rho, lamda/rho)
        u = u + rho * (w-z)
        
        obj_ADMM.append(obj(X, y, w, lamda))
        error_ADMM.append(norm(beta_0 - w))
    return error_ADMM, obj_ADMM, w

## MOM ADMM
def MOM_ADMM(X, y, beta_0, K=3, max_iter=100, lamda = 1):
    dim = X.shape[1]
    x, x_prime = np.matrix([0.0]*dim).T, np.matrix([0.0]*dim).T
    z, z_prime = x, x_prime
    u, u_prime = x, x_prime
    mom_obj_ADMM, mom_error_ADMM = [], []
    rho = 5
    print("Lauching MOM-ADMM solver...")
    for t in range(0, max_iter):
        #if (t%100==0):
        #    print('iter= {}, estimation error MOM ADMM = {:3f}'.format(t, norm(beta_0 - x)))
        #descent
        Xk, yk = block(X, y, x, x_prime, K)
        x = np.linalg.solve((Xk.T)*Xk + rho*np.identity(dim), Xk.T*yk + rho*z - u)
        z = soft_threshod(x + u/rho, lamda/rho)
        u = u + rho * (x-z)
        #ascent
        Xk, yk = block(X, y, x, x_prime, K)
        x_prime = np.linalg.solve((Xk.T)*Xk + rho*np.identity(dim), Xk.T*yk + rho*z_prime - u_prime)
        z_prime = soft_threshod(x_prime + u_prime/rho, lamda/rho)
        u_prime = u_prime + rho * (x_prime-z_prime)

        
        mom_obj_ADMM.append(mom_obj(X, y, x, x_prime, lamda))
        mom_error_ADMM.append(norm(beta_0 - x))   
    return mom_error_ADMM, mom_obj_ADMM, x

## MOM ADMM RANDOM BLOCKS
def MOM_ADMM_RANDOM_BLOCKS(X, y, beta_0, K=3, max_iter=100, lamda = 1):
    dim = X.shape[1]
    x, x_prime = np.matrix([0.0]*dim).T, np.matrix([0.0]*dim).T
    z, z_prime = x, x_prime
    u, u_prime = x, x_prime
    mom_obj_ADMM, mom_error_ADMM, list_ind_selected = [], [], []
    rho = 5
    #print("Lauching MOM-ADMM-RANDOM-BLOCKS solver...")
    for t in range(0, max_iter):
        #if (t%100==0):
        #    print('iter= {}, estimation error MOM ADMM RANDOM BLOCKS = {:3f}'.format(t, norm(beta_0 - x)))
        #descent
        Xk, yk, ind = random_block(X, y, x, x_prime, K)
        x = np.linalg.solve((Xk.T)*Xk + rho*np.identity(dim), Xk.T*yk + rho*z - u)
        z = soft_threshod(x + u/rho, lamda/rho)
        u = u + rho * (x-z)
        list_ind_selected.append(ind)
        #ascent
        Xk, yk, ind = random_block(X, y, x, x_prime, K)
        x_prime = np.linalg.solve((Xk.T)*Xk + rho*np.identity(dim), Xk.T*yk + rho*z_prime - u_prime)
        z_prime = soft_threshod(x_prime + u_prime/rho, lamda/rho)
        u_prime = u_prime + rho * (x_prime-z_prime)
        list_ind_selected.append(ind)

        mom_obj_ADMM.append(mom_obj(X, y, x, x_prime, lamda))
        mom_error_ADMM.append(norm(beta_0 - x))
    return mom_error_ADMM, mom_obj_ADMM, x, list_ind_selected


def cv_data(X, y, V, ind_split = 1):
    N = y.size
    ind = np.arange(ind_split*N//V, (ind_split+1)*N//V)
    return np.delete(X, ind, axis = 0), np.delete(y, ind, axis = 0), X[ind], y[ind]

def block_cv(X_test, y_test, x, K_prime):
    vect_means = []
    N = y_test.size
    K_prime = int(K_prime)
    for k in range(0, K_prime):
        Xk, yk = X_test[k*N//K_prime:(k+1)*N//K_prime], y_test[k*N//K_prime:(k+1)*N//K_prime]
        excess_loss_k = linalg.norm(Xk.dot(x) - yk) ** 2
        vect_means.append(excess_loss_k)
    idx_med = median_index(vect_means)
    #print(idx_med)
    return X_test[idx_med*N//K_prime:(idx_med+1)*N//K_prime], y_test[idx_med*N//K_prime:(idx_med+1)*N//K_prime]

def mom_cv_admm(X, y, beta_0, max_iter, V, K_prime, grid_K, grid_lamda = [1]):
    N = y.size
    cv_error_lamda_K = np.zeros((grid_lamda.size, grid_K.size))
    ind_lamda = 0
    for lamda in grid_lamda:
        ind_K = 0
        for K in grid_K:
            print('----------lamda = {} --------K = {}'.format(lamda, K))
            cv_error = []
            for ind_split in np.arange(0, V):
                X_train, y_train, X_test, y_test = cv_data(X, y, V, ind_split)
                _, _, mom_x_admm = MOM_ADMM(X_train, y_train, beta_0, K, max_iter, lamda)
                Xk_test, yk_test = block_cv(X_test, y_test, mom_x_admm, K_prime)
                r = Xk_test*mom_x_admm - yk_test
                cv_error.append(np.linalg.norm(r))
            cv_error_lamda_K[ind_lamda, ind_K] = np.median(cv_error)
            print('-----cv_error_lamda_K = {}'.format(np.median(cv_error)))
            ind_K = ind_K + 1
        ind_lamda = ind_lamda + 1
    return cv_error_lamda_K

def mom_cv_admm_RANDOM_BLOCKS(X, y, beta_0, max_iter, V, K_prime, grid_K, grid_lamda = [1]):
    N = y.size
    cv_error_lamda_K = np.zeros((grid_lamda.size, grid_K.size))
    ind_lamda = 0
    for lamda in grid_lamda:
        ind_K = 0
        for K in grid_K:
            print('----------lamda = {} --------K = {}'.format(lamda, K))
            cv_error = []
            for ind_split in range(0, V):
                X_train, y_train, X_test, y_test = cv_data(X, y, V, ind_split)
                _, _, mom_x_admm, _ = MOM_ADMM_RANDOM_BLOCKS(X_train, y_train, beta_0, K, max_iter, lamda)
                Xk_test, yk_test = block_cv(X_test, y_test, mom_x_admm, K_prime)
                r = Xk_test*mom_x_admm - yk_test
                #print('------r = {}'.format(r))
                cv_error.append(np.linalg.norm(r))
            cv_error_lamda_K[ind_lamda, ind_K] = np.median(cv_error)
            print('-----cv_error_lamda_K = {}'.format(np.median(cv_error)))
            ind_K = ind_K + 1
        ind_lamda = ind_lamda + 1
    return cv_error_lamda_K

def ind_K_lamda(cv_error_lamda_K, grid_K, grid_lamda):
    ind = np.unravel_index(cv_error_lamda_K.argmin(), cv_error_lamda_K.shape)
    print('---- best choice: lambda={} and K={}'.format(grid_lamda[ind[0]], grid_K[ind[1]]))
    return grid_lamda[ind[0]], grid_K[ind[1]]

def cv_admm(X, y, beta_0, max_iter, V, grid_lamda):
    N = y.size
    cv_error_lamda = []
    for lamda in grid_lamda:
        print('--------------------------------lamda = {}'.format(lamda))
        cv_error = []
        for ind_split in range(0, V):
            X_train, y_train, X_test, y_test = cv_data(X, y, V, ind_split)
            _, _, x_admm = ADMM(X_train, y_train, beta_0, max_iter, lamda)
            r = X_test*x_admm - y_test
            cv_error.append(np.linalg.norm(r))
        cv_error_lamda.append(np.mean(cv_error))
        print('-----cv_error_lamda = {}'.format(np.mean(cv_error_lamda)))
    return np.array(cv_error_lamda)

def ind_lamda(cv_error_lamda, grid_lamda):
    ind = cv_error_lamda.argmin()
    print('---- best choice: lambda={}'.format(grid_lamda[ind]))
    return grid_lamda[ind]

def robustness(X1, y1, beta_0, grid_proportion_outliers, K, lamda):
    error_lasso = []
    error_mom_lasso = []
    for prop in grid_proportion_outliers:
        n_outliers = int(n_samples*prop)
        print('----------------proportion outliers = {} -- n_outliers = {}'.format(prop, n_outliers))
        N = n_samples + n_outliers
        y2, X2 = data2(n_outliers, n_features, type_outliers = 2)
        y, X = data_merge(y1, X1, y2, X2)
        print(np.linalg.norm(y))
        y, X = np.matrix(y).T, np.matrix(X)

        error_ADMM, _, _ = ADMM(X, y, beta_0, max_iter, lamda)
        mom_error_ADMM, _, _ = MOM_ADMM(X, y, beta_0, K, max_iter, lamda)
        error_lasso.append(error_ADMM[-1])
        error_mom_lasso.append(mom_error_ADMM[-1])
    return error_lasso, error_mom_lasso

def robustness_adaptive(X1, y1, beta_0, grid_proportion_outliers, V, grid_K, grid_lamda):
    error_lasso = []
    error_mom_lasso = []
    choice_lambda_lasso = []
    choice_lambda_mom_lasso = []
    choice_K_mom_lasso = []
    for prop in grid_proportion_outliers:
        n_outliers = int(n_samples*prop)
        print('----------------proportion outliers = {} -- n_outliers = {}'.format(prop, n_outliers))
        N = n_samples + n_outliers
        y2, X2 = data4(n_outliers, n_features)
        y, X = data_merge(y1, X1, y2, X2)
        print(np.linalg.norm(y))
        y, X = np.matrix(y).T, np.matrix(X)
        
        perf = cv_admm(X, y, beta_0, max_iter, V, grid_lamda)
        lamda = ind_lamda(perf, grid_lamda)
        choice_lambda_lasso.append(lamda)
        error_ADMM, _, _ = ADMM(X, y, beta_0, max_iter, lamda)
        
        K_prime = int(np.max(grid_K)/V)
        perf = mom_cv_admm(X, y, beta_0, max_iter, V, K_prime, grid_K, grid_lamda)
        lamda, K = ind_K_lamda(perf, grid_K, grid_lamda)
        choice_lambda_mom_lasso.append(lamda)
        choice_K_mom_lasso.append(K)
        mom_error_ADMM, _, _ = MOM_ADMM(X, y, beta_0, K, max_iter, lamda)
        
        error_lasso.append(error_ADMM[-1])
        error_mom_lasso.append(mom_error_ADMM[-1])
    return error_lasso, error_mom_lasso, choice_lambda_lasso, choice_lambda_mom_lasso, choice_K_mom_lasso

def robustness_adaptive_RB(X1, y1, beta_0, grid_proportion_outliers, V, grid_K, grid_lamda):
    error_lasso = []
    error_mom_lasso = []
    choice_lambda_lasso = []
    choice_lambda_mom_lasso = []
    choice_K_mom_lasso = []
    list_ind = []
    for prop in grid_proportion_outliers:
        n_outliers = int(n_samples*prop)
        print('----------------proportion outliers = {} -- n_outliers = {}'.format(prop, n_outliers))
        N = n_samples + n_outliers
        y2, X2 = data4(n_outliers, n_features)
        y, X = data_merge(y1, X1, y2, X2)
        y, X = np.matrix(y).T, np.matrix(X)
        
        perf = cv_admm(X, y, beta_0, max_iter, V, grid_lamda)
        lamda = ind_lamda(perf, grid_lamda)
        choice_lambda_lasso.append(lamda)
        error_ADMM, _, _ = ADMM(X, y, beta_0, max_iter, lamda)
        
        K_prime = np.max(grid_K)//V
        perf = mom_cv_admm_RANDOM_BLOCKS(X, y, beta_0, max_iter, V, K_prime, grid_K, grid_lamda)
        lamda, K = ind_K_lamda(perf, grid_K, grid_lamda)
        choice_lambda_mom_lasso.append(lamda)
        choice_K_mom_lasso.append(K)
        mom_error_ADMM, _, _, list_ind_selected = MOM_ADMM_RANDOM_BLOCKS(X, y, beta_0, K, max_iter, lamda)
        
        error_lasso.append(error_ADMM[-1])
        error_mom_lasso.append(mom_error_ADMM[-1])
        list_ind.append(list_ind_selected)
    return error_lasso, error_mom_lasso, choice_lambda_lasso, choice_lambda_mom_lasso, choice_K_mom_lasso, list_ind





########################################
n_samples, n_features, sparsity, sigma, max_iter = 200, 500, 10, 1, 200

np.random.seed(50)
beta_0 = beta_func(n_features, sparsity)
y1, X1 = data1(n_samples, beta_0,  sigma)
V = 5

beta_0 = np.matrix(beta_0).T

grid_proportion_outliers = np.arange(0, 15, 1)/100
N_max = (1+np.max(grid_proportion_outliers))*n_samples
grid_K =  np.arange(1, N_max//4, 4)
grid_lamda = np.arange(0, 100, 10)/sqrt(n_samples)

#dict_mom_lasso_random_blocks = {}
filename = 'dict_mom_lasso_random_blocks.p'
#for loading
with open(filename, "rb") as f:
    dict_mom_lasso_random_blocks = pickle.load(f)

for step in range(51, 70):
    print('-----------------------------------------------------------------------------step = {}'.format(step))
    now = time.time()
    error_lasso, error_mom_lasso, choice_lambda_lasso, choice_lambda_mom_lasso, choice_K_mom_lasso, list_ind = robustness_adaptive_RB(X1, y1, beta_0, grid_proportion_outliers, V, grid_K, grid_lamda)
    print('-----------------------------------------------------------------------------time = {}'.format(time.time() - now))   
    dict_mom_lasso_random_blocks[step] = [error_lasso, error_mom_lasso, choice_lambda_lasso, choice_lambda_mom_lasso, choice_K_mom_lasso, list_ind]   


#import pickle
#filename = 'dict_mom_lasso_random_blocks.p'
#for saving
with open(filename, "wb") as f:
    pickle.dump(dict_mom_lasso_random_blocks, f)
    










