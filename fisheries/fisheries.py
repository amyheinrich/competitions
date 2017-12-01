import random
import numpy as np
import pandas as pd
import os
import csv
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import (Dense, Dropout, Flatten, Input, Activation)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from PIL import Image

random.seed(1)

########################################################################
#                                                                      #
# Create dataset of greyscale images of size 128x72 from training data #
#                                                                      #
########################################################################

filedir = '.../Kaggle/Fish/train/'
greydir = '.../Kaggle/Fish/train.csv'
greyfile = open(greydir, 'wb')
greywrite = csv.writer(greyfile)

# Create column header for trainfile and write to file
colarr = np.arange(128*72)
cols = ['idnum','label']
cols = cols + ['pixel_' + str(i) for i in colarr]
greywrite.writerows([cols])

# Open training files, convert to pixels and write output
species = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
for fish in species:
    newdir = filedir + fish
    for root, dir, files in os.walk(newdir):
        for f in files:
            if f != 'urls.txt':
                idnum = f.split('.')[0]
                label = fish            
                im = Image.open(newdir + '/' + f).convert('L')
                im_small = im.resize((128,72),Image.ANTIALIAS)
                im_array = np.asarray(im_small) 
                im_flat = np.ravel(im_array)
                newrow = [idnum, label] + list(im_flat)
                greywrite.writerows([newrow])
 
greyfile.close()


########################################################################
#                                                                      #
# Create dataset of original images of size 224x224 from training data #
#                                                                      #
########################################################################

filedir = '.../Kaggle/Fish/train/'
origdir = '.../Kaggle/Fish/train_orig.csv'
origfile = open(origdir, 'wb')
origwrite = csv.writer(origfile)

# Create column header for trainfile and write to file
colarr = np.arange(150528)
cols = ['idnum','label']
cols = cols + ['pixel_' + str(i) for i in colarr]
origwrite.writerows([cols])

# Open training files, convert to pixels and write output
species = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
for fish in species:
    newdir = filedir + fish
    for root, dir, files in os.walk(newdir):
        for f in files:
            if f != 'urls.txt':
                idnum = f.split('.')[0]
                label = fish
                img_path = newdir + '/' + f
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                newrow = [idnum, label] + list(np.ravel(x))
                origwrite.writerows([newrow])
origfile.close()


########################################################################
#                                                                      #
#         Create dataset of VGG16 features from original data          #
#                                                                      #
########################################################################

# Load original images of size 224x224
origdata = pd.read_csv(origdir)

# Load vgg16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
vggdata = base_model.predict(origdata)

# Save features
vggdata.to_csv(filedir + 'vgg.csv')


########################################################################
#                                                                      #
#       Create dataset of VGG16 features from submission data          #
#                                                                      #
########################################################################

# Predict intermediate features on submission data
testdir = '.../Kaggle/Fish/test_stg1/'
test_vggdir = '.../Kaggle/Fish/test_vgg.csv'
test_vggfile = open(test_vggdir, 'wb')
test_vggwrite = csv.writer(test_vggfile)

colarr = np.arange(25088)
cols = ['idnum'] + ['feat_' + str(i) for i in colarr]
test_vggwrite.writerows([cols])

for root, dir, files in os.walk(testdir):
    for f in files:
        if f != 'urls.txt':
            img_path = testdir + f
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = base_model.predict(x)
            newrow = [f] + list(np.ravel(preds))
            test_vggwrite.writerows([newrow])
test_vggfile.close()


########################################################################
#                                                                      #
#       Cluster on greyscale images to find same boat sequences        #
#                                                                      #
########################################################################

# Load greydata
greydata = pd.read_csv(greydir)

# Cluster with KMeans
clust = KMeans(n_clusters=200, random_state=0)
clust.fit(greydata.ix[:,2:])
clustids = clust.labels_
greydata['clustid'] = clustids


########################################################################
#                                                                      #
#       Sample from each cluster/label combination for training        #
#                                                                      #
########################################################################

maxnum = 6 # Maximum number of samples to take from each cluster/label

trainclusters = random.sample(np.arange(200),160)
testclusters = [i for i in np.arange(200) if i not in trainclusters]

trainids = []
for cluster in trainclusters:
    for label in ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']:
        trainfilter = greydata['idnum'][np.logical_and(greydata['clustid']==cluster, 
                                                       greydata['label']==label)]
        if len(trainfilter) > 0:
            samplesize = min(maxnum,len(trainfilter))
            trainsample = trainfilter.sample(n=samplesize, random_state=0)
            trainids = trainids + list(trainsample)

testids = list(greydata['idnum'][greydata['clustid'].isin(testclusters)])


########################################################################
#                                                                      #
#                       Prepare the datasets                           #
#                                                                      #
########################################################################

X_train = vggdata[vggdata['idnum'].isin(trainids)]
X_test = vggdata[vggdata['idnum'].isin(testids)]

y_train = X_train['label']
y_test = X_test['label']

X_train = X_train.drop(['idnum','label'],1)
X_test = X_test.drop(['idnum','label'],1)

X_train = X_train.values.reshape(X_train.shape[0], 7, 7, 512)
X_train = X_train.astype('float32')
X_train = X_train / 255

X_test = X_test.values.reshape(X_test.shape[0], 7, 7, 512)
X_test = X_test.astype('float32')
X_test = X_test / 255

le = LabelEncoder()
y_train_ohe = np_utils.to_categorical(le.fit_transform(y_train))
y_test_ohe = np_utils.to_categorical(le.transform(y_test))


########################################################################
#                                                                      #
#              Prepare data for augmentation                           #
#                                                                      #
########################################################################

# Prepare data for augmentation
X_orig = origdata[origdata['idnum'].isin(trainids)]
y_orig = X_orig['label']

X_orig = X_orig.drop(['idnum','label'],1)
X_orig = X_orig.values.reshape(X_orig.shape[0], 224, 224, 3)
X_orig = X_orig.astype('float32')
X_orig = X_orig / 255

y_orig_ohe = np_utils.to_categorical(le.transform(y_orig))


########################################################################
#                                                                      #
#               Create augmented sample data                           #
#                                                                      #
########################################################################

datagen = ImageDataGenerator(shear_range=0.1,
                             zoom_range=0.1,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True)

datagen.fit(X_orig)

batches = 0

for X_batch, y_batch in datagen.flow(X_orig, y_orig_ohe, batch_size=250, seed=0):
    augment_train = X_batch
    augment_label = y_batch
    batches += 1
    if batches >= 1:
        break
        
# Predict features on augmented data
orig_predict = base_model.predict(augment_train)

# Add augmented data to training data
X_orig = orig_predict.reshape(orig_predict.shape[0], 7, 7, 512)
X_orig = X_orig.astype('float32')
X_orig = X_orig / 255

X_train_full = np.concatenate((X_train,X_orig), axis=0)
y_train_full = np.concatenate((y_train_ohe, augment_label), axis=0)


########################################################################
#                                                                      #
#                Define new top layer to VGG16 model                   #
#                                                                      #
########################################################################

inp = Input(shape=(base_model.output_shape[1:]), name='Input')
x = Flatten(name='Flatten')(inp)
x = Dense(4096, name='Dense_1')(x)
x = Activation('relu', name='Activation_1')(x)
x = Dropout(0.5, name='Dropout_1')(x)
x = Dense(4096, name='Dense_2')(x)
x = Activation('relu', name='Activation_2')(x)
x = Dropout(0.5, name='Dropout_2')(x)
predictions = Dense(8, activation='softmax', name='Dense_output')(x)

top_model = Model(inputs=inp, outputs=predictions)


########################################################################
#                                                                      #
#         Compile and fit model on training + augmented data           #
#                                                                      #
########################################################################

sgd = SGD(lr=.0001, decay=1e-7, momentum=0.7)

# Compile
top_model.compile(loss='categorical_crossentropy', 
                  optimizer='sgd', metrics='accuracy')
                
# Fit
top_model.fit(X_train_full, y_train_full, epochs=50, shuffle=True, batch_size=32, verbose=1,
              validation_data=(X_test,y_test_ohe))
              
# Save top model
top_modelpath = '.../Kaggle/Fish/top_model.h5'
top_weightspath = '.../Kaggle/Fish/top_model_weights.h5'
top_model.save(top_modelpath)
top_model.save_weights(top_weightspath)


########################################################################
#                                                                      #
#                 Predict and format submission data                   #
#                                                                      #
########################################################################

# Load intermediate features
subdata = pd.read_csv(test_vggdir)

# Prepare data
X_sub = subdata.drop('idnum', 1)
X_sub = X_sub.values.reshape(X_sub.shape[0], 7, 7, 512)
X_sub = X_sub.astype('float32')
X_sub = X_sub / 255

# Predict
subpredict = top_model.predict(X_sub)

# Format submission file
header = ['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
submitprobs = pd.DataFrame(subpredict)
submitprobs = pd.concat([subdata[['idnum']],submitprobs], axis=1)
submitprobs.columns = header

# Save submission file
submitprobs.to_csv('.../Kaggle/Fish/submitprobs.csv', index=False)