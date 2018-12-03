import numpy as np
from sklearn.base import BaseEstimator,clone
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
import time
import inspect


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


class perceptronMOM(BaseEstimator):

    '''Perceptron MOM classifier.
    Perceptron MOM risk minimization. The Perceptron minimize the perceptron loss using SGD without regularization.
    
    Parameters
    ----------

    w0 : array-like, length = n_features + 1, default ones(n_features + 1)
        initial coefficients (including the intercept) of the classifier.

    K : int, default 10
        number of blocks for the computation of the MOM. A big value of K deals with more outliers but small values of K are better for the performance when there are no outliers.
        
    eta0 : float, default 1
        step size parameter, the step size is defined as the i-th iteration by 1/(1+eta0*i).

    epoch : int, default 200
        number of iterations before the end of the algorithm.

    mu : float between 0 and 1, default 0.95
        coefficient in the momentum.

    agg : int, default 1
        number of runs of the algorithm on which we aggregate. One might want to decrease this number if the complexity is a problem.

    compter : boolean, default False
        used for outlier detection, if compter=True, the number of time each point is used in the algorithm will be recorded in the attribute "counts".

    progress : boolean, default False
        display a progress bar to monitor the algorithm on each run (agg > 1 means several progress bar).

    verbose : boolean, default True
        display a message at the end of each run if agg > 1.

    multi : {'ovr','ovo'} , default 'ovr'
        method used to go from binary classification to multiclass classification. 'ovr' means "one vs the rest" and 'ovo' means "one vs one" .
        
    Attributes
    ----------
    
    w0 : array like, length = n_features + 1
        w0 is updated in the algorithm, provides with the final coefficients of the decision function.

    counts : array like, length = n_sampled
        the i-th element record the number of time the i-th element of the training dataset X has been used. Only if compter=True.

    Methods
    -------

    fit(X,y) : fit the model
        X : numpy matrix size = (n_samples,n_features)
        y : array like, length = n_samples


    predict(X) : predict the class of the points in X
        X : numpy matrix size = (n_samples,n_features)
        returns array-like, length = n_samples.

    predict_proba(X) : predict the probability that each point belong to each class.
        X : numpy matrox size = (n_samples,n_features)
        returns matrix, size = (n_samples,n_class)
        
    '''

    def __init__( self,w0=None,K=10,eta0=1,epoch=100,mu=0.95,agg=1,compter=False,progress=False, verbose = True, multi='ovr'):
        binary_clf=perceptronMOM_binary(w0,K,eta0,epoch,mu,agg,compter,progress,verbose)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        if multi=="ovr":
            self.clf=OneVsRestClassifier(binary_clf)
        elif multi=="ovo":
            self.clf=OneVsOneClassifier(binary_clf)
        else:
            raise NameError('Multiclass meta-algorithm not known')
    def fit(self,X,y):
        self.clf.fit(X,y)
        return self
    def predict(self,X):
        return self.clf.predict(X)
    def predict_proba(self,X):
        return self.clf.predict_proba(X)
    def score(self,X,y):
        return np.mean(self.predict(X)==y)
    def set_params(self,**params):
        self.__init__(**params)
        return self

class perceptronMOM_binary(BaseEstimator):
    '''Class for algorithm perceptron MOM RM. 
    The loss is max(0,y*f(x)) and f(x)=w^Tx+inter.
    The methods are fit, predict, predict_proba... same idea as in sklearn.
    '''
    def __init__(self,w0=None,K=10,eta0=1,epoch=100,mu=0.95,agg=1,compter=False,progress=False,verbose=True):
        self.w0=w0
        if self.w0 is not None:
            self.coef=w0[:-1]
            self.i0=w0[-1]
        self.K=K
        self.eta0=eta0
        self.epoch=epoch
        self.mu=mu
        self.agg=agg
        self.compter=compter
        self.progress=progress
        self.verbose=verbose

    def fit1(self,X,Y):
        w=np.array(self.coef)
        inter=self.i0
        pas=lambda i : 1/(1+self.eta0*i)**(2/3)
        v=np.zeros(len(X[0]))
        vi=0
        mu=self.mu
        if self.compter:
            self.counts=np.zeros(len(X))
        if self.progress:
            Bar=progressbar(self.epoch)
        for f in range(self.epoch):
            if self.progress:
                Bar.update(f)
            perm=np.random.permutation(len(X))
            X=X[perm]
            Y=Y[perm]
            blocks=blockMOM(self.K,X)
            losses=self.perte(X,Y,w,inter)
            risque,b=MOM(losses,blocks)
            for j in range(len(blocks[b])):
                i=blocks[b][j]
                if ((np.sum(w*X[i])+inter)*Y[i])<0:
                    v=mu*v-pas(f)*Y[i]*X[i]
                    w=w-v
                    vi=mu*vi-pas(f)*Y[i]
                    inter=inter-vi
            if self.compter:
                self.counts[blocks[b]]+=1
        self.w=w
        self.inter=inter
        self.w0=np.hstack([w,inter])
    def fit(self,x,Y):
        if self.w0 is None:
            self.w0=np.zeros(len(x[0])+1)
            self.coef=self.w0[:-1]
            self.i0=self.w0[-1]

        X=np.array(x).copy()
        y=np.array(Y).copy()
        self.values=np.sort(list(set(Y)))
        y[y==self.values[0]]=-1
        y[y==self.values[1]]=1
        w=np.zeros(len(X[0]))

        inter=0
        for f in range(self.agg):
            if self.verbose and self.agg>1:
                print(' Passage ', f)
            self.fit1(X,y)
            w+=self.w
            inter+=self.inter
        self.w=w/self.agg
        self.inter=inter/self.agg
       
    def perte(self,X,y,w,inter):
        pred=(X.dot(w.reshape([len(w),1]))+inter).reshape(len(X))
        result=np.zeros(len(pred))
        result[(-y*pred)>0]=(-y*pred)[(-y*pred)>0]
        return result

    def predict(self,X):
        X=np.array(X).copy()
        pred=(X.dot(self.w.reshape([len(self.w),1]))+self.inter)>=0
        return np.array([self.values[int(p)] for p in pred])
    def decision_function(self,X):
        X=np.array(X).copy()
        pred=(X.dot(self.w.reshape([len(self.w),1]))+self.inter)
        return pred
        
    def score(self,x,y):
        pred=self.predict(x)
        return np.mean(pred==y)

class logregMOM(BaseEstimator):
    '''Logistic Regression MOM classifier.

    Logarithmic regression MOM risk minimization using IRLS with regularization L2.
    
    Parameters
    ----------

    w0 : array-like, length = n_features + 1, default ones(n_features + 1)
        initial coefficients (including the intercept) of the classifier.

    K : int, default 10
        number of blocks for the computation of the MOM. A big value of K deals with more outliers but small values of K are better for the performance when there are no outliers.
        
    eta0 : float, default 1
        step size parameter, the step size is defined as the i-th iteration by 1/(1+eta0*i).

    beta : float, default 1
        L2 regularization parameter.

    epoch : int, default 200
        number of iterations before the end of the algorithm.

    agg : int, default 3
        number of runs of the algorithm on which we aggregate. One might want to decrease this number if the complexity is a problem.

    compter : boolean, default False
        used for outlier detection, if compter=True, the number of time each point is used in the algorithm will be recorded in the attribute "counts".

    progress : boolean, default False
        display a progress bar to monitor the algorithm on each run (agg > 1 means several progress bar).

    verbose : boolean, default True
        display a message at the end of each run if agg > 1.

    multi : {'ovr','ovo'} , default 'ovr'
        method used to go from binary classification to multiclass classification. 'ovr' means "one vs the rest" and 'ovo' means "one vs one" .
        
    Attributes
    ----------
    
    w0 : array like, length = n_features + 1
        w0 is updated in the algorithm, provides with the final coefficients of the decision function.

    counts : array like, length = n_sampled
        the i-th element record the number of time the i-th element of the training dataset X has been used. Only if compter=True.

    Methods
    -------

    fit(X,y) : fit the model
        X : numpy matrix size = (n_samples,n_features)
        y : array like, length = n_samples


    predict(X) : predict the class of the points in X
        X : numpy matrix size = (n_samples,n_features)
        returns array-like, length = n_samples.

    predict_proba(X) : predict the probability that each point belong to each class.
        X : numpy matrox size = (n_samples,n_features)
        returns matrix, size = (n_samples,n_class)
        
    '''
    def __init__(self,w0=None,K=10,eta0=1,beta=1,epoch=200,agg=3,compter=False,progress=False,verbose=True,multi='ovr',augmenter=2,power=2/3):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        binary_clf=logregMOM_binary(w0,K,eta0,beta,epoch,agg,compter,progress,verbose,power)
        if multi=="ovr":
            self.clf=OneVsRestClassifier(binary_clf)
        elif multi=="ovo":
            self.clf=OneVsOneClassifier(binary_clf)
        else:
            #print("Multiclass meta-algorithm not known, choosing 'ovr'")
            self.clf=OneVsRestClassifier(binary_clf)


    def fit(self,X,y):
        perm=np.array([])
        for f in range(self.augmenter):
            perm=np.hstack([perm,np.random.permutation(len(X))])
        perm=perm.astype(np.int64)
        self.clf.fit(X[perm],y[perm])
        return self
    def predict(self,X):
        return self.clf.predict(X)
    def predict_proba(self,X):
        return self.clf.predict_proba(X)
    def score(self,X,y):
        return np.mean(self.predict(X)==y)
    def set_params(self,**params):
        self.__init__(**params)
        return self

class logregMOM_binary(BaseEstimator):
    '''Class of the binary classification for the logistic regression MOM.
    '''
    def __init__(self,w0=None,K=10,eta0=1,beta=1,epoch=200,agg=3,compter=False,progress=False,verbose=True,power=2/3):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    def fit1(self,x,Y):
        w=np.array(self.w0)
        X=np.hstack([np.array(x),np.ones(len(x)).reshape(len(x),1)])

        pas=lambda i : 1/(1+self.eta0*i)**self.power
        if self.compter:
            self.counts=np.zeros(len(X))
        compteur=1
        fincompteur=1
        if self.progress:
            Bar=progressbar(self.epoch)
        for f in range(self.epoch):
            if self.progress:
                Bar.update(f)
            losses=self.perte(X,Y,w)
            blocks=blockMOM(self.K,X)

            compteur+=1
            risque,b=MOM(losses,blocks)
            Xb=X[blocks[b]]
            yb=Y[blocks[b]]
            #IRLS avec regularisation L2
            eta=self.sigmoid(Xb.dot(w.reshape([len(w),1]))).reshape(len(Xb))
            D=np.diag(eta*(1-eta))
            w=w*(1-pas(f))+pas(f)*np.linalg.inv(np.transpose(Xb).dot(D).dot(Xb)+self.beta*np.eye(len(X[0]))).dot(np.transpose(Xb).dot(yb-eta)-self.beta*w)
            if self.compter:
                self.counts[blocks[b]]+=1

            
        return w

    def fit(self,x,Y):
        if self.w0 is None:
            self.w0=np.zeros(len(x[0])+1)
        y=np.array(Y).copy()
        self.values=np.sort(list(set(Y)))
        yj=y.copy()
        indmu=yj!=self.values[1]
        indu=yj==self.values[1]
        yj[indmu]=0
        yj[indu]=1
        w=np.zeros(len(self.w0))
        for f in range(self.agg):
            if self.agg !=1 and self.verbose:
                print('Passage '+str(f))
            w+=self.fit1(x,yj)
        self.w=w/self.agg

    def perte(self,X,y,w):
        pred=X.dot(w.reshape([len(w),1]))
        pred=pred.reshape(len(X))
        return np.log(1+np.exp(-(2*y-1)*pred))

    def predict(self,x):
        X=x.copy
        X=np.hstack([x,np.ones(len(x)).reshape(len(x),1)])

        pred=(X.dot(self.w.reshape([len(self.w),1]))).reshape(len(X))
        return np.array([self.values[int(p>0)] for p in pred])

    def predict_proba(self,x):
        X=x.copy
        X=np.hstack([x,np.ones(len(x)).reshape(len(x),1)])
        pred=self.sigmoid(X.dot(self.w.reshape([len(self.w)])))
        return np.array([[1-p,p] for p in pred])

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def score(self,x,y):
        pred=self.predict(x)
        return np.mean(pred==np.array(y))

