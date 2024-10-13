# -*- coding: utf-8 -*-
"""AE_automated

This file is designed to derive the optimum configuration of a neural network
using Keras hypertune and Keras sequence generators.

"""


import keras_tuner as kt
#from google.colab import drive
import pandas as pd
import nmslib
import mlpack
import time
import glob
#import pdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import subprocess
import h5py
import pdb
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler,QuantileTransformer
import random as python_random
import genieclust
import datetime
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

data_path=r'C:\\Users\\q75714hz\\New folder\\UVLIF\\PLAIR_HK\\Processed\\'
# data_path=r'C:\\Users\\q75714hz\\testhongkong\\'

"""1) Define custom generators that will load the data from multiple CSV files in batches during the training phase. """

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    CONTAINS SPECIFIC INFO FOR AUTOENCODERS
    """
    def __init__(self, list_files, to_fit=True, mini_batch = 1000, batch_size=1, shuffle=False):
        """Initialization
        :param data_path: path to datafiles
        :param list_files: list of image labels (file names)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param shuffle: True to shuffle label indexes after every epoch
        """

        # We have to create a mapping to the file name and the subset of data
        # extracted from that file as a dictionary or list.
        # To do this we need to count the number of lines in each file
        # and then divide that by the mini_batch and loop through each
        # chunck and define a starting point to extract the data

        self.list_files = list_files
        self.mini_batch = mini_batch
        self.data_path = data_path
        #self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        #self.dim = dim
        #self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_files_temp = [self.list_files[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_files_temp)

        #return X

        if self.to_fit:
            y = X
            return (X, y)
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_files_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))


        if len(list_files_temp) == 1:
          path = list_files_temp[0][0]
          start_loc = list_files_temp[0][1]
          end_loc = list_files_temp[0][2]
          stop_point = min(start_loc+self.mini_batch,end_loc)

          #info_df=pd.read_csv(path,skiprows=start_loc+1,nrows=min(self.mini_batch,end_loc-start_loc))
          #Scattering_df = info_df.iloc[:, 34::][(info_df.iloc[:, 34::].T != 0).any()]
          #Scattering_df[Scattering_df < 0] = 0
          #Scattering_df=Scattering_df.div(Scattering_df.max(axis=1), axis=0)
          ## extract the numpy array and then reshape back to the original size.
          #images = Scattering_df.loc[:, Scattering_df.columns != 'label'].to_numpy()
          #X = np.reshape(images, (images.shape[0], 80, 24, 1))
          hf = h5py.File(path, 'r')
          data=hf['test']['block0_values'][start_loc:stop_point, 0:32]

          # data=info_df.iloc[:, 0:32].to_numpy()
          data = data[~np.all(data == 0, axis=1)]
          data[data <= 1] = 1.0
          #X = info_df.iloc[:, 33::].to_numpy().reshape(info_df.shape[0],80,24,1)
          X=np.log10(data) #.reshape(data.shape[0],80,24,1)

        else:
          FL_list=[]
          step=0
          for entry in list_files_temp:
            path = entry[0]
            start_loc = entry[1]
            end_loc = entry[2]
            stop_point = min(start_loc+self.mini_batch,end_loc)
            #info_df=pd.read_csv(path,na_filter=False,header=None,skiprows=start_loc+1,nrows=min(self.mini_batch,end_loc-start_loc))
            #Scattering_df = info_df.iloc[:, 34::][(info_df.iloc[:, 34::].T != 0).any()]
            #Scattering_df[Scattering_df < 0] = 0
            #Scattering_df=Scattering_df.div(Scattering_df.max(axis=1), axis=0)
            #Scattering_df = Scattering_df.dropna()
            hf = h5py.File(path, 'r')
            info_df= hf['test']['block0_values'][start_loc:stop_point, :]
            # info_df=pd.read_hdf(path, "test",start=start_loc,stop=stop_point)
            FL_list.append(info_df)
            step+=1
          FL_df2=pd.concat(FL_list,axis=0) #pd.DataFrame.from_dict(Scattering_dict, orient='index')
          #extract the numpy array and then reshape back to the original size.
          #images = Scattering_df2.loc[:, Scattering_df2.columns != 'label'].to_numpy()
          #pdb.set_trace()
          data=FL_df2.iloc[:, 0:32].to_numpy()
          data = data[~np.all(data == 0, axis=1)]
          #X = Scattering_df2.iloc[:, 33::].to_numpy().reshape(Scattering_df2.shape[0],80,24,1)
          data[data <= 1] = 1.0
          X=np.log10(data)
          X=X.reshape(data.shape[0],32,1)#.reshape(data.shape[0],80,24,1)



        return X

list_files = glob.glob(data_path+'*.hdf')
list_files=sorted(list_files, key=os.path.getsize)

# define a minibatch which would normally be used in the standard training method
minibatch = 1000

list_of_mappings = []
de_all=[]
for filename in list_files:
    print(filename)
    hf=pd.read_hdf(filename,mode='r')
    lines=int(hf.shape[0])
    hf_fluo=hf.iloc[:,0:32]
    de_hf=hf_fluo[np.all(hf_fluo == 0, axis=1)]
    de_all.append(de_hf)
    #pdb.set_trace()
    chunks = int(np.ceil(lines / minibatch))
    for step in range(chunks):
        sublist=[]
        sublist.append(filename)
        sublist.append(step*minibatch)
        sublist.append(min((step + 1)*minibatch,lines))
        list_of_mappings.append(sublist)

training_generator = DataGenerator(list_of_mappings)
validation_generator = DataGenerator(list_of_mappings)
# bilstm_36 = keras.models.load_model(r'C:\Users\q75714hz\Desktop\Classification Research\Hongkong data\encoder_fluo_sequence.h5')
# bilstm_36.summary()
#
# encoder_36 = keras.Model(inputs=bilstm_36.input, outputs=bilstm_36.get_layer('bidirectional_11').output)
# encoder_36.summary()
# encoder_36 = keras.models.load_model(r'C:\Users\q75714hz\Desktop\Classification Research\Hongkong data\encoder_fluo_sequence.h5')
# encoder_36.summary()
# bilstm_36 = keras.models.load_model(r'C:\Users\q75714hz\Desktop\Classification Research\Hongkong data\fluo_hpc_model\ae_fluo_best_change_units_sequencehpc.h5')
# bilstm_36.summary()
# #
# encoder_36 = keras.Model(inputs=bilstm_36.input, outputs=bilstm_36.get_layer('dense_16').output)
# encoder_36.summary()
bilstm_36 = keras.models.load_model(r'C:\Users\q75714hz\Desktop\Classification Research\Hongkong data\fluo_hpc_model\Bilstm_final_final_final_best_fluo_change_units.h5')
bilstm_36.summary()
#
# encoder_36 = keras.Model(inputs=bilstm_36.input, outputs=bilstm_36.get_layer('dense_16').output)
encoder_36 = keras.Model(inputs=bilstm_36.input, outputs=bilstm_36.get_layer('bidirectional_16').output)

encoder_36.summary()


latent_sca=encoder_36.predict(training_generator)
# re_sca=bilstm_36.predict(training_generator)

print(latent_sca.shape)
# pdb.set_trace()

# scaler = StandardScaler()
# scaler.fit(latent_sca)
# processed_scattering_latent = scaler.transform(latent_sca)
scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
processed_scattering_latent = scaler.fit_transform(latent_sca)
# processed_scattering_latent=np.concatenate([processed_scattering_latent,processed_scattering_latent,processed_scattering_latent,processed_scattering_latent],axis=0)
label_list=[]
total_label=pd.DataFrame()
t0 = time.time()
for i in range(2,12):
    ncluster=i
    HCA_model = genieclust.Genie(n_clusters=ncluster,exact=False,gini_threshold=0.3)
    HCA_model.fit_predict(processed_scattering_latent)
    cnn_labels = HCA_model.labels_
    total_label['label'+str(i)]=cnn_labels
print("time elapsed - first run: %.3f" % (time.time() - t0))
    # label_list.append(cnn_labels)
# total_label=pd.concat(label_list,axis=1)
# total_label=pd.DataFrame(total_label)

total_label.to_hdf('C:\\Users\\q75714hz\\Desktop\\re-label\\'+'bi_fluo_new_quantran_genis_label.hdf',key='label_value')
