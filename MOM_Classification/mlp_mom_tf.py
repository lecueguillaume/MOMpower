import numpy as np
from scipy.misc import imread
import tensorflow as tf
import os
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

class mlp_MOM_image():

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

    epoch_count : int, default 50
        say to display the risque every epoch_count epochs.
    
    num_classes : int, default None
        number of class, automatically computed in training but needed if we only use prediction using a saved neural netowrk.

    Attributes
    ----------
    

    Methods
    -------
    
    fit(input_files_train,y_train)
        input_files_train : list of string or string or matrix
            contain the training sample, can be under the form of filepath or a matrix
        y_train : array-like, length = n_test_sample
        compute the weights and bias  of the neural network. Export the weights and bias in the file self.save_file

    predict(input_files_test)
        input_files_train : list of string or string or matrix
            contain the test sample, can be under the form of a list of filepaths (string) or a regex string path. 
        compute the predicted labels from the test sample using the weights and bias from save_file (can be used without using fit)

    predict_proba(input_files_test)
        input_files_train : list of string or string or matrix
            contain the test sample, can be under the form of a list of filepaths (string) or a regex string path.
        compute the predicted probabilities from the test sample using the weights and bias from save_file (can be used without using fit)


    '''

    def __init__(self,shape=[100],K=10,learning_rate=1e-3,beta=1e-4,epochs=100,batch_size = 1000,stddev=0.01,regex=False,save_file='graph_mlp.tf',epoch_count=50,num_classes=None):
        self.learning_rate=learning_rate
        self.beta=beta
        self.epochs=epochs
        self.batch_size=batch_size
        self.K=K
        self.shape=shape
        self.stddev=stddev
        self.regex=regex
        self.save_file=save_file
        self.epoch_count=epoch_count
        self.num_classes=num_classes
        self.values=[]
        
    def fit(self,input_files_train,y_train):
        """
        If self.data_images=True then input_files_train and input_files_test are expected to be a list of filepaths or a regex expression (in that last case, also set self.regex=True).
        else, the input_files_train and input_files_test are supposed to be the 2D matrices of the samples.
        y_train is supposed to be a 1D array containing the labels.
        """

        tf.reset_default_graph()
        gen_tensor=self.distort_input(input_files_train)
        if self.regex:
            filepaths=glob.glob(input_files_train)
        else:
            filepaths=input_files_train
        image1=imread(filepaths[0])
        n_init=self.IMAGE_SIZE**2*3
        self.n_init=n_init
        x = tf.placeholder(tf.float32, [None,n_init])
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
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #print(tf.all_variables())
        saver = tf.train.Saver(W+b)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)

            # Training
            a=time.time()
             
            for epoch in range(self.epochs):
                generator_x=self.generate_image(input_files_train,self.batch_size)
                cost = []
                batch=0
                for images in generator_x:
                    batch_x=sess.run(gen_tensor,feed_dict={self.image: images})
                    if ((batch+1)*self.batch_size) < len(labels):
                        batch_y=labels[(self.batch_size*batch):(self.batch_size*(batch+1))]
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
                if epoch % self.epoch_count ==0:
                    print("Epoch:", (epoch+1 ), "cost =", "{:.3f}".format(np.mean(cost)),' en environ ',(time.time()-a), 's')

            saver.save(sess, self.save_file);
            print('Weights saved in '+self.save_file)
        tf.reset_default_graph()

    def predict_proba(self,input_files_test):
        tf.reset_default_graph()
        gen_tensor=self.distort_input(input_files_test)

        if self.regex:
            filepaths=glob.glob(input_files_test)
        else:
            filepaths=input_files_test
        image1=imread(filepaths[0])
        self.n_init=self.IMAGE_SIZE**2*3

        x = tf.placeholder(tf.float32, [None,self.n_init ])
        y = tf.placeholder(tf.float32, [None, self.num_classes])
        K=tf.constant(self.K)
        shapes=[(self.n_init,self.shape[0])]+[(self.shape[f],self.shape[f+1]) for f in range(len(self.shape)-1)]+[(self.shape[-1],self.num_classes)]
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

        #Prediction
        saver = tf.train.Saver(W+b)

        images=self.generate_image(input_files_test,None,True)
        with tf.Session() as sess:
            saver.restore(sess,self.save_file)
            for image in images:
                xtest=sess.run(gen_tensor,feed_dict={self.image: image})

                pred=sess.run(y_clipped,feed_dict={x: xtest})
            return pred
        tf.reset_default_graph()

    def predict(self,input_files_test):
        pred=self.predict_proba(input_files_test)
        
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
        if len(self.values)==0:
            return [self.values[v] for v in y]
        else:
            return [v for v in y]


    def distort_input(self,images_filepath):
        if self.regex:
            filepaths=glob.glob(images_filepath)
        else:
            filepaths=images_filepath
        image1 = imread(images_filepath[0]).astype(np.float)

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        self.im_h=np.shape(image1)[0]
        self.im_w=np.shape(image1)[1]
        self.IMAGE_SIZE=int(np.min([np.shape(image1)[0],np.shape(image1)[1]])*0.9)
        IMAGE_SIZE=self.IMAGE_SIZE
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        filenames = []
        idx = 0
        self.image= tf.placeholder(tf.float32, [None,np.shape(image1)[0],np.shape(image1)[1],3])
        distorted_image = tf.map_fn(lambda img: tf.random_crop(img, [height, width,3]),self.image)

        # Randomly flip the image horizontally.
        distorted_image = tf.map_fn(lambda dist_img : tf.image.random_flip_left_right(dist_img),distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.map_fn(lambda dist_img: tf.image.random_brightness(dist_img,max_delta=63),distorted_image)
        distorted_image = tf.map_fn(lambda dist_img: tf.image.random_contrast(dist_img,
                                                 lower=0.2, upper=1.8),distorted_image)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.map_fn(lambda dist_img: tf.image.per_image_standardization(dist_img),distorted_image)

        images=tf.contrib.layers.flatten(float_image)
        
        return images
    def generate_image(self,input_files,batch_size=None,test=False):
        if self.regex:
            filepaths=glob.glob(input_files)
        else:
            filepaths=input_files
        if test:
            batch_size=len(filepaths)
        images=np.zeros([np.min([len(filepaths),batch_size]),self.im_h,self.im_w,3])

        idx=0
        batch=0
        for filepath in filepaths:
            image = imread(filepath).astype(np.float)
            if len(np.shape(image))==3:
                images[idx,:,:,:]=image
                idx += 1
            else:
                if test:
                    print('one of the test sample is not 3D')
                    images[idx,:,:,0]=image
                    images[idx,:,:,1]=image
                    images[idx,:,:,2]=image
            if idx == batch_size:
                yield  images
                batch+=1
                images=np.zeros([np.min([len(filepaths)-batch*batch_size,batch_size]),self.im_h,self.im_w,3])
                idx = 0
        if idx > 0:
            yield  images




