
from PIL import Image
import tensorflow as tf
import cifar10
import numpy as np
import glob
width = 50
height = 50

import pandas as pd
df=pd.read_csv('dataset/names.csv')
categories=np.array(df['names'])

#filenames = glob.glob("dataset/test_resized/*.jpg") # absolute path to input images
filenames = glob.glob("dataset/eval/*.jpg") # absolute path to input images
result=[]

fn=tf.placeholder(tf.string)
input_img = tf.image.decode_jpeg(tf.read_file(fn), channels=3)
tf_cast = tf.cast(input_img, tf.float32)
float_image = tf.image.resize_image_with_crop_or_pad(tf_cast, height, width)
float_image=tf.image.per_image_standardization(float_image)
images = tf.expand_dims(float_image, 0)
logits = cifar10.inference(images)
_, top_k_pred = tf.nn.top_k(logits, k=1)
init_op = tf.initialize_all_variables()
res=[]
with tf.Session() as sess:
 # Restore variables from training checkpoint.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.get_checkpoint_state('cifar10_train')
    if ckpt and ckpt.model_checkpoint_path:
        print("ckpt.model_checkpoint_path ", ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
        exit(0)
    sess.run(init_op)

    for filename in filenames:
        a, top_indices = sess.run([_, top_k_pred],feed_dict={fn:filename})
#        for key, value in enumerate(top_indices[0]):
#            print (categories[value] + ", " + str(a[0][key]))
        res+=[categories[top_indices[0][0]]] 


df2=pd.read_csv('dataset/evaly.csv')
y=np.array(df2['y'])

print(np.mean(np.array(res)==y))


