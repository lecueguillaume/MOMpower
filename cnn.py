import numpy as np
from scipy.misc import imread
import tensorflow as tf
import os
import time
import glob



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

def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_'+name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class CNN():

    '''CNN classifier.
    Convolutional neural network 
    
    Parameters
    ----------

    channels : list of ints, length = n_layers, default [64,32],

    filter_shape : list of ints, length = 2, default [5,5],
        shape of the filter for the convolution layers.

    pool_shape ; list of ints, length = 2, default [2,2],
        shape of the pooling filter for the pooling layers.
    
    final_layer_shape : int, default 1000,
        length of the final connected layer.

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

    def __init__(self,channels=[32,64],filter_shape=[5,5],pool_shape=[2,2],final_layer_shape=1000,learning_rate=1e-3,beta=1e-4,epochs=100,batch_size = 1000,stddev=0.01,regex=False,save_file='graph_cnn.tf',epoch_count=50,num_classes=None,tensorboard_file='tensorboard_logs'):
        self.learning_rate=learning_rate
        self.beta=beta
        self.epochs=epochs
        self.batch_size=batch_size
        self.channels=channels
        self.pool_shape=pool_shape
        self.filter_shape=filter_shape
        self.final_layer_shape=final_layer_shape
        self.stddev=stddev
        self.regex=regex
        self.save_file=save_file
        self.epoch_count=epoch_count
        self.num_classes=num_classes
        self.values=[]
        self.tb_file=tensorboard_file
        
    def fit(self,input_files_train,y_train):
        """
        If self.data_images=True then input_files_train and input_files_test are expected to be a list of filepaths or a regex expression (in that last case, also set self.regex=True).
        else, the input_files_train and input_files_test are supposed to be the 2D matrices of the samples.
        y_train is supposed to be a 1D array containing the labels.
        """

        if os.path.exists(self.tb_file+"/CNN"):
            for filename in glob.glob(self.tb_file+'/CNN/*'):
                os.remove(filename)
            os.rmdir(self.tb_file+'/CNN')
       

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
        x_shaped = tf.reshape(x, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
        
        labels=self.one_hot(y_train)
        y = tf.placeholder(tf.float32, [None, self.num_classes])
        shapes=[(self.channels[f],self.channels[f+1]) for f in range(len(self.channels)-1)]

        layer,w,b=self.create_new_conv_layer(x_shaped,3,self.channels[0],self.filter_shape,self.pool_shape,name='layer1')
        L=[layer]
        W=[w]
        B=[b]
        i=0
        for f,g in shapes:
            i+=1
            layer,w,b=self.create_new_conv_layer(L[-1],f,g,self.filter_shape,self.pool_shape,name='layer'+str(i))
            L+= [layer]
            W+=[w]
            B+=[b]
            variable_summaries(w,'Conv_weight_'+str(i))
            variable_summaries(b,'Conv_bias_'+str(i))
        
        flattened=tf.contrib.layers.flatten(L[-1])

        #n=tf.shape(flattened)[1]
        n=self.IMAGE_SIZE
        for f in range(len(self.channels)):
            n=np.ceil(n/2)
        
        n=int(self.channels[-1]*n**2)
        wf1 = tf.Variable(tf.truncated_normal([n, self.final_layer_shape], stddev=self.stddev), name='wf1')
        bf1 = tf.Variable(tf.truncated_normal([self.final_layer_shape], stddev=self.stddev), name='bf1')
        variable_summaries(wf1,'dense_weight_1')
        variable_summaries(bf1,'dense_bias_1')
        dense_layer1 = tf.matmul(flattened, wf1) + bf1
        dense_layer1 = tf.nn.relu(dense_layer1)

        wf2 = tf.Variable(tf.truncated_normal([self.final_layer_shape, self.num_classes], stddev=self.stddev), name='wf2')
        variable_summaries(wf2,'dense_weight_2')
        bf2 = tf.Variable(tf.truncated_normal([self.num_classes], stddev=self.stddev), name='bf2')
        variable_summaries(bf2,'dense_bias_2')
        dense_layer2 = tf.matmul(dense_layer1, wf2) + bf2

        y_ = tf.nn.softmax(dense_layer2)
        perte = tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y)
        for i in range(len(W)):
            perte=tf.add(perte,self.beta*tf.nn.l2_loss(W[i]))

        cross_entropy = tf.reduce_mean(perte)
        tf.summary.scalar('Cross_entropy', cross_entropy)


        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)
        #print(tf.all_variables())
        saver = tf.train.Saver(W+B+[wf1,wf2,bf1,bf2])
        init_op = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.tb_file+'/CNN',
                                      sess.graph)
            sess.run(init_op)
            # Training
            a=time.time()
            generator_x=self.generate_image(input_files_train,self.batch_size)
            for epoch in range(self.epochs):
                
                cost = []
                batch=0
                
                Xs=[]
                losses=[]
                indices=[]
                images,inds=generator_x.__next__()
                batch_x=sess.run(gen_tensor,feed_dict={self.image: images})
                batch_y=labels[inds]
                
                _, c,summary = sess.run([optimiser, cross_entropy,merged],
                                  feed_dict={x: batch_x, y: batch_y})
                train_writer.add_summary(summary, epoch)

                cost += [c]
                if epoch % self.epoch_count ==0:
                    print("Epoch:", (epoch+1 ), "cost =", "{:.3f}".format(np.mean(cost)),' en environ ',(time.time()-a),'s')

            saver.save(sess, self.save_file);
            print('Weights saved in '+self.save_file)
        tf.reset_default_graph()

    def create_new_conv_layer(self,input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                          num_filters]

        # initialise weights and bias for the filter
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=self.stddev),
                                          name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)

        # now perform max pooling
        ksize = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                                   padding='SAME')

        return out_layer,weights,bias

    def predict_proba(self,input_files_test):


        tf.reset_default_graph()
        gen_tensor=self.distort_input(input_files_test,True)

        if self.regex:
            filepaths=glob.glob(input_files_test)
        else:
            filepaths=input_files_test
        image1=imread(filepaths[0])
        self.n_init=self.IMAGE_SIZE**2*3
        
        x = tf.placeholder(tf.float32, [None,self.n_init])
        x_shaped = tf.reshape(x, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
        
        y = tf.placeholder(tf.float32, [None, self.num_classes])
        shapes=[(self.channels[f],self.channels[f+1]) for f in range(len(self.channels)-1)]

        layer,w,b=self.create_new_conv_layer(x_shaped,3,self.channels[0],self.filter_shape,self.pool_shape,name='layer1')
        L=[layer]
        W=[w]
        B=[b]
        i=0
        for f,g in shapes:
            i+=1
            layer,w,b=self.create_new_conv_layer(L[-1],f,g,self.filter_shape,self.pool_shape,name='layer'+str(i))
            L+= [layer]
            W+=[w]
            B+=[b]
          
        flattened=tf.contrib.layers.flatten(L[-1])
        
        n=self.IMAGE_SIZE
        for f in range(len(self.channels)):
            n=np.ceil(n/2)
        
        n=int(self.channels[-1]*n**2)
        
        wf1 = tf.Variable(tf.truncated_normal([n, self.final_layer_shape], stddev=self.stddev), name='wf1')
        bf1 = tf.Variable(tf.truncated_normal([self.final_layer_shape], stddev=self.stddev), name='bf1')
        dense_layer1 = tf.matmul(flattened, wf1) + bf1
        dense_layer1 = tf.nn.relu(dense_layer1)

        wf2 = tf.Variable(tf.truncated_normal([self.final_layer_shape, self.num_classes], stddev=self.stddev), name='wf2')
        bf2 = tf.Variable(tf.truncated_normal([self.num_classes], stddev=self.stddev), name='bf2')
        dense_layer2 = tf.matmul(dense_layer1, wf2) + bf2

        y_ = tf.nn.softmax(dense_layer2)

        #Prediction
        saver = tf.train.Saver(W+B+[wf1,wf2,bf1,bf2])

        images=self.generate_image(input_files_test,None,True)
        with tf.Session() as sess:
            saver.restore(sess,self.save_file)
            for image in images:
                xtest=sess.run(gen_tensor,feed_dict={self.image: image})

                pred=sess.run(y_,feed_dict={x: xtest})
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


    def distort_input(self,images_filepath,test=False):
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
        

        if not test:
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
        else:

            self.image= tf.placeholder(tf.float32, [None,np.shape(image1)[0],np.shape(image1)[1],3])
            images=tf.image.resize_images(self.image,[height,width])

            images = tf.map_fn(lambda img: tf.image.per_image_standardization(img),images)
            images=tf.contrib.layers.flatten(images)

            return images
    def generate_image(self,input_files,batch_size=None,test=False):
        
        if self.regex:
            filepaths=glob.glob(input_files)
        else:
            filepaths=input_files
        if test:
            batch_size=len(filepaths)

        idx=0
        if not test:
            inds=[]
            while True:
                images=np.zeros([batch_size,self.im_h,self.im_w,3])
                for _ in range(batch_size):
                    ind=int(np.random.uniform()*len(filepaths))
                    inds+=[ind]
                    image = imread(filepaths[ind]).astype(np.float)
                    if len(np.shape(image))==3:
                        images[idx,:,:,:]=image
                        idx += 1
                    else:
                        if test:
                            print('one of the test sample is not 3D')
                            images[idx,:,:,0]=image
                            images[idx,:,:,1]=image
                            images[idx,:,:,2]=image
                        idx+=1
                    if idx == batch_size:
                        yield  images,inds
                        idx = 0
                        inds=[]
        else:
            images=np.zeros([len(filepaths),self.im_h,self.im_w,3])
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

            yield  images




