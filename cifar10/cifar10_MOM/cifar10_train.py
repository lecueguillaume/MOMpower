# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")

import os
import pandas
K=3
import numpy as np


def MOM(x):

    means_blocks=[np.mean(xx ) for xx in x]
    indice=np.argsort(means_blocks)[int(np.floor(len(means_blocks)/2))]
    return indice

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.

    
    images2=tf.placeholder(tf.float32,(FLAGS.batch_size,24,24,3),name='input_images')
    labels2=tf.placeholder(tf.int32,(FLAGS.batch_size),name='input_labels')

    logits2=cifar10.inference(images2)
    loss=cifar10.loss(logits2,labels2)
    train_op = cifar10.train(loss, global_step)


    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference_reuse(images)

    # Calculate loss.
    losses = cifar10.loss1(logits, labels)


    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()
        if os.path.isfile('log_cifar.csv'): 
          os.remove('log_cifar.csv')
        df=pandas.DataFrame([],columns=['time','step','loss'])
        df.to_csv('log_cifar.csv',index=False)
      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))



          df = pandas.read_csv('log_cifar.csv')
          df2 = pandas.DataFrame([[time.time(), self._step ,  loss_value ]], columns=['time','step','loss'])    
          df=df.append(df2)
          df.to_csv('log_cifar.csv',index=False)

    with tf.train.MonitoredTrainingSession(
            save_checkpoint_secs=300,
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      imagemom=np.zeros([FLAGS.batch_size,24,24,3])
      lblmom=np.zeros(FLAGS.batch_size)
      cost=mon_sess.run([loss],feed_dict={images2:imagemom, labels2:lblmom})
      while not mon_sess.should_stop():
        lossbs=[]
        ibs=[]
        lblbs=[]
        for k in range(K):
            lossb,ib,lblb=mon_sess.run([losses,images,labels],feed_dict={images2:imagemom, labels2:lblmom })
            lossbs+=[lossb]
            ibs+=[ib]
            lblbs+=[lblb]
        ind=MOM(lossbs)
        imagemom=ibs[ind]
        lblmom=lblbs[ind]
        mon_sess.run([train_op],feed_dict={images2:imagemom, labels2:lblmom})


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
