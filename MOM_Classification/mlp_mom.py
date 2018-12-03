import numpy as np
from sklearn.base import BaseEstimator,clone
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import log_loss
import time


class progressbar():
    '''Just a simple progress bar.
    '''
    def __init__(self,N):
        self.N=N
    def update(self,i):
        percent=int((i+1)/self.N*100)
        if i != self.N-1:
            print('\r'+"["+"-"*percent+' '*(100-percent)+']', end='')
        else:
            print('\r'+"["+"-"*percent+' '*(100-percent)+']')

def blockMOM(K,x):
    '''Sample the indices of K blocks for data x using a random permutation

    Parameters
    ----------

    K : int
        number of blocks

    x : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do blocks for.

    Returns 
    -------

    list of size K containing the lists of the indices of the blocks, the size of the lists are contained in [n_sample/K,2n_sample/K]
    '''
    b=int(np.floor(len(x)/K))
    nb=K-(len(x)-b*K)
    nbpu=len(x)-b*K
    perm=np.random.permutation(len(x))
    blocks=[[(b+1)*g+f for f in range(b+1) ] for g in range(nbpu)]
    blocks+=[[nbpu*(b+1)+b*g+f for f in range(b)] for g in range(nb)]
    return [perm[b] for  b in blocks]

def MOM(x,blocks):
    '''Compute the median of means of x using the blocks blocks

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    blocks : list of list, provided by the function blockMOM.

    Return
    ------

    The median of means of x using the block blocks, a float.
    '''
    means_blocks=[np.mean([ x[f] for f in ind]) for ind in blocks]
    indice=np.argsort(means_blocks)[int(np.ceil(len(means_blocks)/2))]
    return means_blocks[indice],indice

class mlp_MOM(BaseEstimator):

    '''MLP MOM classifier.
    Multi layer perceptron MOM risk minimization. MLP is a neural network that minimizes the log loss.
    
    Parameters
    ----------

    shape : tuple, length = n_layers -2, default (100,)
        the i-th element represent the number of neurons in the i-th hidden layer.

    K : int, default 10
        number of blocks for the computation of the MOM. A big value of K deals with more outliers but small values of K are better for the performance when there are no outliers.
        
    eta0 : float, default 1
        step size parameter, the step size is defined as the i-th iteration by 1/(1+eta0*i).

    beta : float, default 1
        L2 regularization parameter.

    epoch : int, default 100
        number of iterations before the end of the algorithm.

    agg : int, default 1
        number of runs of the algorithm on which we aggregate. One might want to decrease this number if the complexity is a problem.

    progress : boolean, default False
        display a progress bar to monitor the algorithm on each run (agg > 1 means several progress bar).

    verbose : boolean, default True
        display a message at the end of each run if agg > 1.

    Attributes
    ----------
    
    Same as the attributes of MLPClassifier class in sklearn

    Methods
    -------
    
    Same as the attributes of MLPClassifier class in sklearn

    '''

    def __init__(self,shape=(100,),K=100,eta0=1,beta=0.0001,epoch=100,agg=1,progress=False,verbose=True):
        self.shape=shape
        self.K=K
        self.eta0=eta0
        self.beta=beta
        self.epoch=epoch
        self.agg=agg
        self.progress=progress
        self.verbose=verbose

    def fit1(self,x,Y,clf,coefs,intercepts):
        pas=lambda i : 1/(1+self.eta0*i)
        clfi=clone(clf)
        if self.progress:
            Bar=progressbar(self.epoch)
        for f in range(self.epoch):
            if self.progress:
                Bar.update(f)
            blocks=blockMOM(self.K,x)
            losses=self.perte(x,Y,clf)
            risque,b=MOM(losses,blocks)
            Xb=x[blocks[b]]
            yb=Y[blocks[b]]
            clfi.partial_fit(Xb,yb,classes=self.classes)
        return clfi

    def fit(self,x,Y):
        self.classes=np.sort(list(set(Y)))
        x=np.array(x).copy()
        y=np.array(Y).copy()
        for f in range(len(self.classes)):
            y[y==self.classes[f]]=f

        clf=MLPClassifier(hidden_layer_sizes=self.shape,alpha=self.beta,learning_rate_init=self.eta0)
        clf.partial_fit(x,Y,classes=self.classes)
        coefs=clf.coefs_
        intercepts=clf.intercepts_
        coefs_tot=[np.zeros(np.shape(c)) for c in coefs]
        intercepts_tot=[np.zeros(np.shape(c)) for c in intercepts]

        for f in range(self.agg):
            if self.verbose and (self.agg>1):
                print('Passage '+str(f))
            clfi=self.fit1(x,Y,clf,coefs,intercepts)
            for i in range(len(coefs)):
                coefs_tot[i]+=clfi.coefs_[i]
                intercepts_tot[i]+=clfi.intercepts_[i]
        coefs=[c/self.agg for c in coefs_tot]
        intercepts=[c/self.agg for c in intercepts_tot]
        clf.coefs_=coefs
        clf.intercepts_=intercepts
        self.clf=clf

    def getloss(self):
            return lambda x : np.log2(1 + np.exp(-x))

    def perte(self,X,y,clf):
        pred=clf.predict_proba(X)
        basis=np.eye(len(self.classes))
        return np.array([log_loss(np.array(basis[self.classes==y[i]]),np.array([pred[i]])) for i in range(len(y))])
    def predict(self,x):
        return self.clf.predict(x)

    def predict_proba(self,x):
        return self.clf.predict_proba(x)

    def score(self,x,y):
        pred=self.predict(x)
        return np.mean(pred==np.array(y))
