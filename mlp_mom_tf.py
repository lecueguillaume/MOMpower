import numpy as np
from scipy.misc import imread
import tensorflow as tf
import os


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

class mlp_MOM():

    '''MLP MOM classifier.
    Multi layer perceptron MOM risk minimization. MLP is a neural network that minimizes the log loss.
    
    Parameters
    ----------

    shape : list, length = n_layers -2, default (100,)
        the i-th element represent the number of neurons in the i-th hidden layer.

    K : int, default 10
        number of blocks for the computation of the MOM. A big value of K deals with more outliers but small values of K are better for the performance when there are no outliers.
        
    learning_rate : float, default 1e-3
        step size parameter, the step size is defined as the i-th iteration by 1/(1+eta0*i).

    beta : float, default 1e-4
        L2 regularization parameter.

    epochs : int, default 100
        number of iterations before the end of the algorithm.

    batch_size : int, default 1000
        size of a batch, one should not choose a batch_size to small because the effective batch size will be batch_size/K
    stddev : float, default 0.1
        std of the normal initialization of the weights and bias.
    
    regex : boolean, default False
        whether or not the filepath are given using regex or a list

    progress : boolean, default False
        display a progress bar to monitor the algorithm on each run (agg > 1 means several progress bar).

    data_images : boolean, default True
        whether or not the data are images.

    Attributes
    ----------
    
    Same as the attributes of MLPClassifier class in sklearn

    Methods
    -------
    
    Same as the attributes of MLPClassifier class in sklearn

    '''

    def __init__(self,shape=[100],K=10,learning_rate=1e-3,beta=1e-4,epochs=100,batch_size = 1000,stddev=0.1,regex=False,progress=False,data_images=True):
        self.learning_rate=learning_rate
        self.beta=beta
        self.epochs=epochs
        self.batch_size=batch_size
        self.K=K
        self.shape=shape
        self.stddev=stddev
        self.regex=regex
        self.progress=progress
        self.data_images=data_images
        
    def fit_predict(self,input_files_train,y_train,input_files_test):
        """
        If self.data_images=True then input_files_train and input_files_test are expected to be a list of filepaths or a regex expression (in that last case, also set self.regex=True).
        else, the input_files_train and input_files_test are supposed to be the 2D matrices of the samples.
        y_train is supposed to be a 1D array containing the labels.
        """
        if self.data_images:
            if self.regex:
                filepaths=glob.glob(input_files_train)
            else:
                filepaths=input_files_train
            image1=imread(filepaths[0])
            n_init=np.shape(image1)[0]*np.shape(image1)[1]
        else:
            n_init= len(input_files_train[0])
        x = tf.placeholder(tf.float32, [None,n_init ])
        K=tf.constant(self.K)
        labels=self.one_hot(y_train)
        y = tf.placeholder(tf.float32, [None, self.num_classes])
        shapes=[(n_init,self.shape[0])]+[(self.shape[f],self.shape[f+1]) for f in range(len(self.shape)-1)]+[(self.shape[-1],self.num_classes)]
        W=[]
        b=[]
        for i in range(len(shapes)):
            f,g = shapes[i]
            W += [tf.Variable(tf.random_normal([f, g], stddev=self.stddev), name='W'+str(i))]
            b += [tf.Variable(tf.random_normal([g]), name='b'+str(i))]
        h =[tf.add(tf.matmul(x, W[0]), b[0])]
        H = [tf.nn.relu(h[-1])]
        for i in range(len(self.shape)-1):
            h +=[tf.add(tf.matmul(H[i], W[i+1]), b[i+1])]
            H += [tf.nn.relu(h[i+1])]
        y_ = tf.nn.softmax(tf.add(tf.matmul(H[-1], W[-1]), b[-1]))
        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        perte = -tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1)
        
        for i in range(len(W)):
            perte=tf.add(perte,self.beta*tf.nn.l2_loss(W[i]))
        cross_entropy=tf.reduce_mean(perte)
     
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)
        init_op = tf.global_variables_initializer()
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        with tf.Session() as sess:

            # Training
             
            sess.run(init_op)
            if self.progress:
                Bar=progressbar(self.epochs)
            for epoch in range(self.epochs):
                if self.progress:
                    Bar.update(epoch)
                cost = []
                batch=0
                if self.data_images:
                    generator_x=self.load_images(input_files_train,self.batch_size)
                else:
                    generator_x=[(0,input_files_train)]
                for filename,images in generator_x:
                    batch_x= images
                    if ((batch+1)*self.batch_size) < len(labels):
                        batch_y=labels[(self.batch_size*batch):(self.batch_size*batch+1)]
                    else:
                        batch_y=labels[(self.batch_size*batch):]
                    blocks=blockMOM(self.K,batch_y)
                    losses=sess.run(perte,feed_dict={x: batch_x,y: batch_y})
                    
                    risk,block_MOM=MOM(losses,blocks)
                    batch+=1
                    batch_x= np.array(batch_x)[blocks[block_MOM]]
                    batch_y=np.array(batch_y)[blocks[block_MOM] ]

                    _, c = sess.run([optimiser, cross_entropy],
                                  feed_dict={x: batch_x, y: batch_y})
                    cost += [c]
                if epoch % 50 ==0:
                    print("Epoch:", (epoch ), "cost =", "{:.3f}".format(np.mean(cost)))

            #Prediction
            if self.data_images:
                generator_x=self.load_images(input_files_test,test=True)
            else:
                generator_x=[(0,input_files_test)]
            
            for f,xtest in generator_x:
                pred=sess.run(y_clipped,feed_dict={x: xtest})
        return self.one_hot_reverse(pred)

    def load_images(self,input_files, batch_size=None,test=False):
        """Read png/jpg images from input files in batches.

        Args:
        input_files: input directory, regex or list of filepaths
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

        Yields:
        filenames: list file names without path of each image
          Lenght of this list could be less than batch_size, in this case only
          first few images of the result are elements of the minibatch.
        images: array with all images from this batch
        """
        if self.regex:
            filepaths=glob.glob(input_files)
        else:
            filepaths=input_files
        if test:
            batch_size=len(filepaths)
        image1=imread(filepaths[0])

        images = np.zeros([np.min([batch_size,len(filepaths)]),np.shape(image1)[0]*np.shape(image1)[1]])
        filenames = []
        idx = 0

        for filepath in filepaths:
            image = imread(filepath).astype(np.float) 
            im= 0.2126*image[:,:,0]+0.7152*image[:,:,1]+0.0722*image[:,:,2]

            images[idx, :]=im.reshape([np.shape(image1)[0]*np.shape(image1)[1]])
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, images
                filenames = []
                images = np.zeros([np.min([batch_size,len(filepaths)]),np.shape(image1)[0]*np.shape(image1)[1]])
                idx = 0
        if idx > 0:
            yield filenames, images
    
    def one_hot(self,y):
        self.values,inv=np.unique(y,return_inverse=True)
        self.num_classes=len(self.values)
        labels=np.zeros([len(y),self.num_classes])
        for l in range(len(y)):
            labels[l][inv[l]]+=1
        return labels
    def one_hot_reverse(self,labels):
        y=np.argmax(labels,axis=1)
        return [self.values[v] for v in y]


