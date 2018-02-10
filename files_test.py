import os
import numpy as np
from PIL import Image
from sklearn.cross_validation import train_test_split

def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist


path2 = './leapmotion_dataset'
imlist = modlistdir(path2)
    
image1 = np.array(Image.open(path2 +'/' + imlist[0])) 

m,n = image1.shape[0:2] 
total_images = len(imlist) 

immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                        for images in imlist], dtype = 'f')



print (immatrix.shape)

label=np.ones((total_images,),dtype = int)

samples_per_class = int(total_images / nb_classes)
print "samples_per_class - ",samples_per_class
s = 0
r = int(samples_per_class)
for classIndex in range(nb_classes):
    label[s:r] = classIndex
    s = r
    r = s + samples_per_class


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]
    
(X, y) = (train_data[0],train_data[1])
    
    
# Split X and y into training and testing sets
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
    
# normalize
X_train /= 255
X_test /= 255
    
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
return X_train, X_test, Y_train, Y_test










