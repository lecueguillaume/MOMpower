
CIFAR-10 README :

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/

--------------------------------------------------
I did a fork of this algorithm that use mom risk minimization and that can be applied on any image dataset. To test this the file "test_tiny.py" will download the dataset tiny imagenet, prepare it and then you can train the model on this dataset using the program "cifar10_train.py" (the first time, say no when asked whether or not you want to load saved weights).
Additionally to the initial code, the dataset is split into two part : one for train and one for evaluation. You can use "evaluation.py" to get the accuracy on the evaluation dataset using the neural network saved in the last checkpoint. Checkpoint are automatically saved by the algorithm at regular intervals and if you have already began to train the model, you can continue by reloading the checkpoint at the beginning of the training by saying "y" when asked and by indicating the directory in which the checkpoints are (usually cifar10_train).
The architecture of the NN is fixed in cifar10.py and can be changed in that file, all the images are resized to 64x64 then randomly croped to 50x50.
