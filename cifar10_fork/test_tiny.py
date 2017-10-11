#exemple d'utilisation avec le dataset tiny_imagenet
url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
from six.moves import urllib
import sys
import zipfile
import os
import glob
import numpy as np
def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % ("dataset/tiny.zip",
      float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
filepath, _ = urllib.request.urlretrieve(url, "dataset/tiny.zip", _progress)
print()
statinfo = os.stat(filepath)

zip_ref=zipfile.ZipFile("dataset/tiny.zip", 'r')
zip_ref.extractall('dataset')
zip_ref.close()

train_files=glob.glob('dataset/tiny-imagenet-200/train/*/images/*.JPEG')
import pandas as pd
import shutil
df=pd.DataFrame()
ids=[]
labels=[]
for filepath in train_files:
    name=filepath.split('/')[-1]
    name=name.split('.')[0]
    os.rename(filepath, "dataset/train/"+name+'.jpg')
    ids+=[name]
    labels+=[filepath.split('/')[-3]]
df['id']=ids
df['labels']=labels
df.to_csv('dataset/labels.csv')

test_files=glob.glob('dataset/tiny-imagenet-200/test/images/*.JPEG')
ids=[]
for filepath in test_files:
    name=filepath.split('/')[-1]
    name=name.split('.')[0]
    os.rename(filepath, "dataset/test/"+name+'.jpg')
    ids+=[name]

df2=pd.DataFrame()
df2['id']=ids
df2['class']=np.zeros(len(df2))
df2.to_csv('dataset/sample_submission.csv')
shutil.rmtree('dataset/tiny-imagenet-200')
#On fait que tester et donc faire une évaluation
print('dataset créé, formattage pour que les fichiers soient comme il faut.')

os.system("python3 prepare_cifar10.py")

print("Plus qu'à lancer le training avec la commande python3 cifar10_train.py (ceci n'est pas fait automatiquement pour éviter de faire ramer l'ordi si l'on est en train de faire quelque chose)")
