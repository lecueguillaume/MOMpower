import numpy as np
from sklearn.base import BaseEstimator,clone
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel
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

class log_kernel_MOM(BaseEstimator):
    ''' Logistic Regression Kernel MOM
    
    Kernel logarithmic regression MOM risk minimization using IRLS with regularization L2
    
    Parameters
    ----------

    K : int, default 10
        number of blocks for the computation of the MOM. A big value of K deals with more outliers but small values of K are better for the performance when there are no outliers.
        
    eta0 : float, default 1
        step size parameter, the step size is defined as the i-th iteration by 1/(1+eta0*i).

    beta : float, default 1
        L2 regularization parameter.

    epoch : int, default 200
        number of iterations before the end of the algorithm.

    kernel : {'rbf','poly', callable function}, default 'rbf'
        kernel used in the algorithm. A callable function can be given, it should take as entry two matrices X1, X2 and return the pairwise kernel distance matrix 

    gamma : float, default 1/n_features
        coefficient used if the kernel is 'rbf' in which case the kernel function is exp(-gamma*x^2)

    degree : int, default 3
        degree of the polynomial if the kernel is 'poly'

    agg : int, default 1
        number of runs of the algorithm on which we aggregate. One might want to decrease this number if the complexity is a problem.

    verbose : boolean, default True
        display a message at the end of each run if agg > 1.

    progress : boolean, default False
        display a progress bar to monitor the algorithm on each run (agg > 1 means several progress bar).

    compter : boolean, default False
        used for outlier detection, if compter=True, the number of time each point is used in the algorithm will be recorded in the attribute "counts".

    multi : {'ovr','ovo'} , default 'ovr'
        method used to go from binary classification to multiclass classification. 'ovr' means "one vs the rest" and 'ovo' means "one vs one" .
        
    Attributes
    ----------
    
    alpha : array like, length = n_sample
        alpha is updated in the algorithm, provides with the final coefficients of the decision function.

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
    def __init__(self,K=10,eta0=1,beta=1,epoch=200,kernel='rbf',gamma=None,degree=3,agg=1,verbose=True,progress=False,compter=False,multi='ovr'):
        binary_clf=log_kernel_MOM_binary(K,eta0,beta,epoch,kernel,gamma,degree,agg,verbose,progress,compter)
        if multi=="ovr":
            self.clf=OneVsRestClassifier(binary_clf)
        elif multi=="ovo":
            self.clf=OneVsOneClassifier(binary_clf)
        else:
            raise NameError('Multiclass meta-algorithm not known')
    def fit(self,X,y):
        self.clf.fit(X,y)
    def predict(self,X):
        return self.clf.predict(X)
    def predict_proba(X):
        return self.clf.predict_proba(X)


class log_k():
    '''Class of KLR
    '''
    def __init__(self,y,l):
        self.y=y.copy()
        self.y[y==0]=-1
        self.l=l

    def solveWKRR(self,W,z,K):

        sqrtW=np.diag(np.sqrt(W))
        return sqrtW.dot(np.linalg.inv(sqrtW.dot(K).dot(sqrtW)+ self.l*len(W)*np.eye(len(W)))).dot(sqrtW.dot(z))

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def irlsk(self,b,blocks,alpha_0,pas,K):
        Kb=K[blocks[b]][:,blocks[b]]
        alpha=alpha_0[blocks[b]]
        m=Kb.dot(alpha)
        P=-self.sigmoid(-self.y[blocks[b]]*m)
        W=self.sigmoid(m)*self.sigmoid(-m)+1e-10
        z=m-P*self.y[blocks[b]]/W
        alpha=alpha_0[blocks[b]]+pas*(self.solveWKRR(W,z,Kb)-alpha_0[blocks[b]])
        alphaf=alpha_0.copy()
        alphaf[blocks[b]]=alpha
        return alphaf


class log_kernel_MOM_binary(BaseEstimator):
    def __init__(self,K=10,eta0=1,beta=1,epoch=200,kernel='rbf',gamma=None,degree=3,agg=1,verbose=True,progress=False,compter=False):
       self.K=K
       self.eta0=eta0
       self.beta=beta
       self.epoch=epoch
       self.gamma=gamma
       self.agg=agg
       self.kernel=kernel
       self.degree=degree
       self.verbose=verbose
       self.progress=progress
       self.compter=compter
       
    def fit1(self,X,y,Kernel):
        self.X=np.array(X)
        X=self.X
        pas=lambda i : 1/(1+self.eta0*i)
        clf=log_k(y,self.beta)
        if self.progress:
            Bar=progressbar(self.epoch)
        if self.compter:
            self.counts=np.zeros(len(X))
        alpha=np.zeros(len(X))
        res=[]
        for f in range(self.epoch):
            if self.progress:
                Bar.update(f)
            blocks=blockMOM(self.K,X)
            losses=self.perte(Kernel,y,alpha)
            risque,b=MOM(losses,blocks)
            alpha=clf.irlsk(b,blocks,alpha,pas(f),Kernel)
            self.alpha=alpha
            if self.compter:
                self.counts[blocks[b]]+=1
        pred=Kernel.dot(self.alpha)
        self.separatrice=np.mean([np.median(pred[y==1]),np.median(pred[y!=1])])
        
    def fit(self,X,Y):
        if (self.kernel=='poly'):
           kfunc=lambda x,y : polynomial_kernel(x,y,degree=self.degree,gamma=self.gamma)
        elif(self.kernel=='rbf'):
           kfunc=lambda x,y : rbf_kernel(x,y,self.gamma)
        else :
           kfunc=self.kernel
        Kernel=kfunc(np.array(X),np.array(X))

        self.values=np.sort(list(set(Y)))
        yj=np.array(Y).copy()
        indmu=yj!=self.values[1]
        indu=yj==self.values[1]
        yj[indmu]=0
        yj[indu]=1

        alpha=np.zeros(len(X))
        for f in range(self.agg):
            if self.verbose and self.agg != 1:
                print('Passage '+str(f))
            self.fit1(X,yj,Kernel)
            alpha+=self.alpha
        self.alpha=alpha/self.agg

    def perte(self,K,y,alpha):
        Kalpha=K.dot(alpha)
        return np.log(1+np.exp(-(2*np.array(y)-1)*Kalpha))+self.beta/2*np.sum(alpha*Kalpha)
   
    def predict(self,xtest):
        
        if (self.kernel=='poly'):
           kfunc=lambda x,y : polynomial_kernel(x,y,degree=self.degree,gamma=self.gamma)
        elif(self.kernel=='rbf'):
           kfunc=lambda x,y : rbf_kernel(x,y,self.gamma)
        else :
           kfunc=self.kernel
        KC=kfunc(xtest,self.X)
        pred=(np.floor(KC.dot(self.alpha)>=self.separatrice)).reshape(len(xtest))
        return np.array([self.values[int(p)] for p in pred])

    def predict_proba(self,x):
        if (self.kernel=='poly'):
           kfunc=lambda x,y : polynomial_kernel(x,y,degree=self.degree,gamma=self.gamma)
        elif(self.kernel=='rbf'):
           kfunc=lambda x,y : rbf_kernel(x,y,self.gamma)
        else :
           kfunc=self.kernel
        KC=kfunc(x,self.X)
        pred=KC.dot(self.alpha)-self.separatrice
        pred=1/(1+np.exp(-pred))
        return np.array([[1-p,p] for p in pred])


class log_kernel_MOM_fast(BaseEstimator):
    ''' Fast Logistic Regression Kernel MOM
    
    Fast Kernel logarithmic regression MOM risk minimization using IRLS with regularization L2
    
    Parameters
    ----------

    K : int, default 10
        number of blocks for the computation of the MOM. A big value of K deals with more outliers but small values of K are better for the performance when there are no outliers.
        
    eta0 : float, default 1
        step size parameter, the step size is defined as the i-th iteration by 1/(1+eta0*i).

    beta : float, default 1
        L2 regularization parameter.

    epoch : int, default 200
        number of iterations before the end of the algorithm.

    kernel : {'rbf','poly', callable function}, default 'rbf'
        kernel used in the algorithm. A callable function can be given, it should take as entry two matrices X1, X2 and return the pairwise kernel distance matrix 

    gamma : float, default 1/n_features
        coefficient used if the kernel is 'rbf' in which case the kernel function is exp(-gamma*x^2)

    degree : int, default 3
        degree of the polynomial if the kernel is 'poly'

    agg : int, default 1
        number of runs of the algorithm on which we aggregate. One might want to decrease this number if the complexity is a problem.

    verbose : boolean, default True
        display a message at the end of each run if agg > 1.

    progress : boolean, default False
        display a progress bar to monitor the algorithm on each run (agg > 1 means several progress bar).

    compter : boolean, default False
        used for outlier detection, if compter=True, the number of time each point is used in the algorithm will be recorded in the attribute "counts".

    multi : {'ovr','ovo'} , default 'ovr'
        method used to go from binary classification to multiclass classification. 'ovr' means "one vs the rest" and 'ovo' means "one vs one" .
        
    Attributes
    ----------
    
    alpha : array like, length = n_sample
        alpha is updated in the algorithm, provides with the final coefficients of the decision function.

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



    def __init__(self,K=10,eta0=1,beta=1,epoch=300,kernel='rbf',gamma=None,degree=3,agg=1,progress=False,multi='ovr'):
        binary_clf=log_kernel_MOM_fast_binary(K,eta0,beta,epoch,kernel,gamma,degree,agg,progress)
        if multi=="ovr":
            self.clf=OneVsRestClassifier(binary_clf)
        elif multi=="ovo":
            self.clf=OneVsOneClassifier(binary_clf)
        else:
            raise NameError('Multiclass meta-algorithm not known')
    def fit(self,X,y):
        self.clf.fit(X,y)
    def predict(self,X):
        return self.clf.predict(X)
    def predict_proba(X):
        return self.clf.predict_proba(X)

class log_k2():
    '''Class of KLR
    '''
    def __init__(self,y,l):
        self.y=y.copy()
        self.y[y==0]=-1
        self.l=l

    def solveWKRR(self,W,z,K):
        sqrtW=np.diag(np.sqrt(W))
        return sqrtW.dot(np.linalg.inv(sqrtW.dot(K).dot(sqrtW)+ self.l*len(W)*np.eye(len(W)))).dot(sqrtW.dot(z))

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def irlsk(self,b,blocks,alpha_0,pas,Kblocks):
        Kb=Kblocks[b]
        alpha=alpha_0[blocks[b]]
        m=Kb.dot(alpha)
        P=-self.sigmoid(-self.y[blocks[b]]*m)
        W=self.sigmoid(m)*self.sigmoid(-m)
        z=m-P*self.y[blocks[b]]/W
        alpha=alpha_0[blocks[b]]+pas*(self.solveWKRR(W,z,Kb)-alpha_0[blocks[b]])
        alphaf=alpha_0.copy()
        alphaf[blocks[b]]=alpha
        return alphaf

class log_kernel_MOM_fast_binary(BaseEstimator):
    def __init__(self,K=10,eta0=1,beta=1,epoch=300,kernel='rbf',gamma=None,degree=3,agg=1,progress=False):
       self.K=K
       self.eta0=eta0
       self.beta=beta
       self.epoch=epoch
       self.gamma=gamma
       self.agg=agg
       self.kernel=kernel
       self.degree=degree
       self.progress=False
       
    def fit1(self,X,y,Kernelblocks):
        self.X=np.array(X).copy()
        X=self.X
        pas=lambda i : 1/(1+self.eta0*i)
        clf=log_k2(y,self.beta)
        
        alpha=np.zeros(len(X))
        res=[]
        if self.progress:
            Bar=progressbar(self.epoch)
        for f in range(self.epoch):
            if self.progress:
                Bar.update(f)
            losses=self.perte(Kernelblocks,y,alpha)
            risque,b=MOM(losses,self.blocks)
            alpha=clf.irlsk(b,self.blocks,alpha,pas(f),Kernelblocks)
            self.b=b
            self.alpha=alpha
        pred=self.decision_function(X)
        self.separatrice=np.mean([np.median(pred[y==1]),np.median(pred[y!=1])])


    def fit(self,X,Y):
        if (self.kernel=='poly'):
           kfunc=lambda x,y : polynomial_kernel(x,y,degree=self.degree,gamma=self.gamma)
        elif(self.kernel=='rbf'):
           kfunc=lambda x,y : rbf_kernel(x,y,self.gamma)
        else :
           kfunc=self.kernel

        self.values=np.sort(list(set(Y)))
        yj=np.array(Y).copy()
        indmu=yj!=self.values[1]
        indu=yj==self.values[1]
        yj[indmu]=0
        yj[indu]=1

        X=np.array(X).copy()
        blocks=blockMOM(int(self.K),X)
        self.blocks=blocks
        Kernelblocks=[]
        for b in range(len(blocks)):
            Kernelblocks+=[kfunc(np.array(X[blocks[b]]),np.array(X[blocks[b]]))]
        alpha=np.zeros(len(X))
        for f in range(self.agg):
            if self.agg != 1:
                print('Passage '+str(f))
            self.fit1(X,yj,Kernelblocks)
            alpha+=self.alpha
        self.alpha=alpha/self.agg

    def perte(self,Kblocks,y,alpha):
        Kalpha=np.zeros(len(alpha))
        for b in range(len(self.blocks)):
            for i in range(len(self.blocks[b])):
                Kalpha[self.blocks[b][i]]=np.sum(Kblocks[b][i]*alpha[self.blocks[b]])
        return np.log(1+np.exp(-(2*y-1)*Kalpha))+self.beta/2*np.sum(alpha*Kalpha)

    def predict(self,xtest):
        
        if (self.kernel=='poly'):
           kfunc=lambda x,y : polynomial_kernel(x,y,degree=self.degree,gamma=self.gamma)
        elif(self.kernel=='rbf'):
           kfunc=lambda x,y : rbf_kernel(x,y,self.gamma)
        else :
           kfunc=self.kernel
        KC=kfunc(xtest,self.X[self.blocks[self.b]])
        pred=(np.floor((KC.dot(self.alpha[self.blocks[self.b]]))>=self.separatrice)).reshape(len(xtest))
        return np.array([self.values[int(p)] for p in pred])

    def predict_proba(self,x):
        pred=self.decision_function(x)-self.separatrice
        pred=1/(1+np.exp(-pred))
        return np.array([[1-p,p] for p in pred])

    def decision_function(self,x):
        if (self.kernel=='poly'):
           kfunc=lambda x,y : polynomial_kernel(x,y,degree=self.degree,gamma=self.gamma)
        elif(self.kernel=='rbf'):
           kfunc=lambda x,y : rbf_kernel(x,y,self.gamma)
        else :
           kfunc=self.kernel
        KC=kfunc(x,self.X[self.blocks[self.b]])
        pred=KC.dot(self.alpha[self.blocks[self.b]]).reshape(len(x))
        return pred

    def score(self,x,y):
        pred=self.predict(x)
        return np.mean(pred==y)
