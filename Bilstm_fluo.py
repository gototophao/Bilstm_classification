# -*- coding: utf-8 -*-
"""AE_automated

This file is designed to derive the optimum configuration of a neural network
using Keras hypertune and Keras sequence generators.

"""


import keras_tuner as kt
#from google.colab import drive
import pandas as pd
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


# data_path=r'C:\\Users\\q75714hz\\New folder\\UVLIF\\PLAIR_HK\\Processed\\'
# data_path=r'C:\\Users\\q75714hz\\testhongkong\\'
data_path='/nobackup/projects/bdman08/PLAIR_HK/Processed/'

"""1) Define custom generators that will load the data from multiple CSV files in batches during the training phase. """

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    CONTAINS SPECIFIC INFO FOR AUTOENCODERS
    """
    def __init__(self, list_files, to_fit=True, mini_batch = 2000, batch_size=1, shuffle=True):
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

# define a minibatch which would normally be used in the standard training method
minibatch = 2000

list_of_mappings = []

for filename in list_files:
    hf=pd.read_hdf(filename,mode='r')
    lines=int(hf.shape[0])
    #pdb.set_trace()
    chunks = int(np.ceil(lines / minibatch))
    for step in range(chunks):
        sublist=[]
        sublist.append(filename)
        sublist.append(step*minibatch)
        sublist.append(min((step + 1)*minibatch,lines-2))
        list_of_mappings.append(sublist)

training_generator = DataGenerator(list_of_mappings)
validation_generator = DataGenerator(list_of_mappings)
print(len(list_of_mappings[:]))
print(list_of_mappings[0:10])



def model_builder(hp):
    number_layers = hp.Int('layers', min_value=3, max_value=4, step=1)
    if number_layers == 3:
        intermediate_filter1_hp_units = hp.Int('units1', min_value=22, max_value=30, step=2)
        intermediate_filter2_hp_units = hp.Int('units2', min_value=12, max_value=22, step=2)
        intermediate_filter3_hp_units = hp.Int('units3', min_value=4, max_value=12, step=2)
        dropout_rate=hp.Choice('dropout_rate',values=[0.1, 0.15,0.2],default=0.1)

        #intermediate_filter3_hp_units = hp.Int('units3', min_value=12, max_value=24, step=2)
        timesteps=32
        n_features=1
        #init = tf.keras.initializers.VarianceScaling(scale=variance_scale, mode='fan_in',distribution='uniform')

        original_inputs = tf.keras.Input(shape=(timesteps,1), name="lstm_encoder_input")

        layer1_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter1_hp_units,return_sequences=True))(original_inputs)
        layer1_lstm = layers.Dropout(rate=dropout_rate)(layer1_lstm)
        layer2_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter2_hp_units,return_sequences=True))(layer1_lstm)
        layer2_lstm = layers.Dropout(rate=dropout_rate)(layer2_lstm)
        layer3_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter3_hp_units, return_sequences=False))(layer2_lstm)
        #encoder_lstm = tf.keras.Model(inputs=original_inputs, outputs=layer2_lstm, name="encoder_lstm")
        #latent_inputs_lstm = tf.keras.Input(shape=(filter2_units), name="lstm_latent_input")
        dec_layer1_lstm = layers.RepeatVector(timesteps)(layer3_lstm)
        dec_layer2_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter3_hp_units, return_sequences=True))(dec_layer1_lstm)
        dec_layer3_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter2_hp_units, return_sequences=True))(dec_layer2_lstm)
        dec_layer4_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter1_hp_units, return_sequences=True))(dec_layer3_lstm)
        decoder_output = layers.TimeDistributed(layers.Dense(n_features))(dec_layer4_lstm)

        lstm_ae = tf.keras.Model(inputs=original_inputs, outputs=decoder_output, name="lstm_ae")

    else:
        intermediate_filter1_hp_units = hp.Int('units1', min_value=22, max_value=30, step=2)
        intermediate_filter2_hp_units = hp.Int('units2', min_value=12, max_value=22, step=2)
        intermediate_filter3_hp_units = hp.Int('units3', min_value=8, max_value=12, step=2)
        intermediate_filter4_hp_units = hp.Int('units4', min_value=2, max_value=12, step=2)
        dropout_rate = hp.Choice('dropout_rate', values=[0.1, 0.15, 0.2], default=0.1)

        # intermediate_filter3_hp_units = hp.Int('units3', min_value=12, max_value=24, step=2)
        timesteps = 32
        n_features = 1
        # init = tf.keras.initializers.VarianceScaling(scale=variance_scale, mode='fan_in',distribution='uniform')

        original_inputs = tf.keras.Input(shape=(timesteps, 1), name="lstm_encoder_input")

        layer1_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter1_hp_units, return_sequences=True))(
            original_inputs)
        layer1_lstm = layers.Dropout(rate=dropout_rate)(layer1_lstm)
        layer2_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter2_hp_units, return_sequences=True))(
            layer1_lstm)
        layer2_lstm = layers.Dropout(rate=dropout_rate)(layer2_lstm)
        layer3_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter3_hp_units, return_sequences=True))(
            layer2_lstm)
        layer3_lstm = layers.Dropout(rate=dropout_rate)(layer3_lstm)
        layer4_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter4_hp_units, return_sequences=False))(
            layer3_lstm)

        # encoder_lstm = tf.keras.Model(inputs=original_inputs, outputs=layer2_lstm, name="encoder_lstm")
        # latent_inputs_lstm = tf.keras.Input(shape=(filter2_units), name="lstm_latent_input")
        dec_layer1_lstm = layers.RepeatVector(timesteps)(layer4_lstm)
        dec_layer2_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter4_hp_units, return_sequences=True))(
            dec_layer1_lstm)
        dec_layer3_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter3_hp_units, return_sequences=True))(
            dec_layer2_lstm)
        dec_layer4_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter2_hp_units, return_sequences=True))(
            dec_layer3_lstm)
        dec_layer5_lstm = layers.Bidirectional(layers.LSTM(intermediate_filter1_hp_units, return_sequences=True))(
            dec_layer4_lstm)
        decoder_output = layers.TimeDistributed(layers.Dense(n_features))(dec_layer5_lstm)

        lstm_ae = tf.keras.Model(inputs=original_inputs, outputs=decoder_output, name="lstm_ae")


    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    lstm_ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])

    return lstm_ae

tuner = kt.Hyperband(model_builder,
                     objective='loss',
                     max_epochs=7,
                     factor=3,
                     directory='/nobackup/projects/bdman08/hao_model_results/my_dir_bilstm_fluo',
                     project_name='intro_to_kt',overwrite=True)
#objective='val_loss',

test_df=pd.read_hdf(list_files[0], "test",start=50,stop=10000)
#extract the numpy array and then reshape back to the original size.
#images = Scattering_df2.loc[:, Scattering_df2.columns != 'label'].to_numpy()
#pdb.set_trace()
X = test_df.iloc[:, 33::].to_numpy().reshape(test_df.shape[0],80,24,1)

tuner.search(training_generator, epochs=16, workers=7)
#tuner.search(x=X,y=X, validation_data=(X,X), epochs=10, workers=4)
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

#print(f"""
#The hyperparameter search is complete. The optimal number of filters in the first Conv layer is {best_hps.get('units1')}.
#The optimal number of filters in the second Conv layer is {best_hps.get('units2')},
#the optimal number of latent units is {best_hps.get('latent_units')},
#optimal dense activation {best_hps.get('dense_activation')}, optimal variance_scale {best_hps.get('variance_scale')},
#optimal pooling choice is {best_hps.get('pooling_choice')},
#and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
#""")

#pdb.set_trace()

"""3) Build the model with the optimal hyperparameters and train it on the data for 20 epochs. """

model = tuner.hypermodel.build(best_hps)
history = model.fit(training_generator, epochs=60,workers=7)
model.save('/nobackup/projects/bdman08/hao_model_results/Bilstm_fluo.h5')
val_acc_per_epoch = history.history['loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
history_new = hypermodel.fit(training_generator, epochs=best_epoch,workers=7)
hypermodel.save('/nobackup/projects/bdman08/hao_model_results/Bilstm_best_fluo.h5')
# plot loss history
# loss = history_new.history['loss']
# #val_loss = history_new.history['val_loss']
# epochs = range(len(loss))
#plt.figure()
#plt.plot(epochs, loss, 'bo', label='Training loss')
##plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training loss')
#plt.legend()
#plt.show()

# pdb.set_trace()


"""5) Save the models. """
# pdb.set_trace()


# Save the encoder
#encoder_cnn.save('/home/dave/Documents/data/UVLIF/encoder_CNN_AE/')