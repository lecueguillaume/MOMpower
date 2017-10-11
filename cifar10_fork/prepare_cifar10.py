import numpy as np
from scipy.misc import imread,imsave
from scipy.misc import imresize
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pickle
from PIL import Image
import os
from array import *


"""
Fichier qui prépare la base de donnée pour avoir le bon format. 
les labels du training sont dans dataset/labels.csv (le label est appelé ici "breed"
un exemple de soumission (pour avoir les id du test set) sont dans dataset/sample_submission.csv
les images train sont dans dataset/train/ et finissent pas .jpg
les images de test sont dans dataset/test et finissent pas .jpg
les images sont toutes resized vers du 64x64
1/10 de la base de training est mise de côté pour permettre une evaluation après la phase de training.
avant de lancer ce programme il faut créer deux dossiers : dataset/eval et dataset/test_resized pour stocker les image remise à l'échelle
Si une des variables (nom de dossier ou tailler d'image) change il faut aussi le changer dans les fichiers cifar10*.py
"""




labels=pd.read_csv('dataset/labels.csv')
ids_train=np.array(labels['id'])
train_files=['dataset/train/'+id+'.jpg' for id in ids_train]
ytrain=np.array(labels['labels'])

model_test=pd.read_csv('dataset/sample_submission.csv')
ids_test=np.array(model_test['id'])
test_files=['dataset/test/'+id+'.jpg' for id in ids_test]

label_names=np.unique(ytrain)
def transform(x):
    return np.array(range(len(label_names)))[label_names==x][0]
ytrain_num=[transform(x) for x in ytrain]
size=(64,64,3)
data = array('B')

train=range(int(len(train_files)*9/10))
evaluation=range(int(len(train_files)*9/10),len(train_files))

eval_files=np.array(train_files)[evaluation]
eval_y=np.array(ytrain)[evaluation]
train_files=np.array(train_files)[train]
ytrain=np.array(ytrain_num)[train]

ids_eval=np.array(ids_train)[evaluation]
ids_train=np.array(ids_train)[train]

for f in range(len(train_files)):
    filename=train_files[f]
    im=imread(filename)
    im=imresize(im,size)
    
    class_name = ytrain[f]
    data.append(class_name)
    if len(np.shape(im))<3:
        print('une des images est en noir et blanc')
 
        for color in range(0,3):
            for x in range(0,64):
                for y in range(0,64):
                    data.append(im[x,y])

    else:
        for color in range(0,3):
            for x in range(0,64):
                for y in range(0,64):
                    data.append(im[x,y,color])



    
    
output_file = open('dataset/train.bin', 'wb')
data.tofile(output_file)
output_file.close()
data = array('B')

for f in range(len(eval_files)):
    filename=eval_files[f]
    im=imread(filename)
    im=imresize(im,size)
    
    class_name = ytrain[f]

    data.append(class_name)
    if len(np.shape(im))<3:
        print('une des images est en noir et blanc')
 
        for color in range(0,3):
            for x in range(0,64):
                for y in range(0,64):
                    data.append(im[x,y])

    else:
        for color in range(0,3):
            for x in range(0,64):
                for y in range(0,64):
                    data.append(im[x,y,color])
  


for f in range(len(eval_files)):
    filename=eval_files[f]
    im=imread(filename)
    im=imresize(im,size)
    imsave("dataset/eval/"+ids_eval[f]+'.jpg',im)

for f in range(len(test_files)):
    filename=test_files[f]
    im=imread(filename)
    im=imresize(im,size)
    imsave("dataset/test_resized/"+ids_test[f]+'.jpg',im)
    
df=pd.DataFrame()
df['y']=eval_y
df.to_csv('dataset/evaly.csv')


df=pd.DataFrame()
df['names']=label_names
df.to_csv('dataset/names.csv')
