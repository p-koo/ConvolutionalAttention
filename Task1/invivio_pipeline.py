import numpy as np
import requests as rq
import io, h5py, os
from six.moves import cPickle

import tensorflow as tf
from tfomics import moana, evaluate
import models
import utils
import argparse

#-----------------------------------------------------------------


def load_basset_data(filepath, reverse_compliment=False):
    trainmat = h5py.File(filepath, 'r')
    x_train = np.array(trainmat['train_in']).astype(np.float32)
    y_train = np.array(trainmat['train_out']).astype(np.int32)
    x_valid = np.array(trainmat['valid_in']).astype(np.float32)
    y_valid = np.array(trainmat['valid_out']).astype(np.int32)
    x_test = np.array(trainmat['test_in']).astype(np.float32)
    y_test = np.array(trainmat['test_out']).astype(np.int32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]

        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])

        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])
        
    return x_train, y_train, x_valid, y_valid, x_test, y_test



def load_deepsea_data(file_path, reverse_compliment=False):
    dataset = h5py.File(file_path, 'r')
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.float32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.float32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)

    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]
        
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])
        
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test

#-----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, help="dataset")
parser.add_argument("-m", type=str, default=0.05, help="model_name")
parser.add_argument("-p", type=int, default=20, help="pool_size")
parser.add_argument("-a", type=str, default='relu', help="activation")
parser.add_argument("-f", type=int, default=64, help="activation")
parser.add_argument("-t", type=int, default=None, help="trial")
args = parser.parse_args()

dataset = args.d
model_name = args.m
pool_size = args.p
activation = args.a
trial = args.t
num_filters = args.f

# set paths
results_path = '../results'
if not os.path.exists(results_path):
    os.makedirs(results_path)
results_path = os.path.join(results_path, dataset)
if not os.path.exists(results_path):
    os.makedirs(results_path)

# load data
data_path = '../../data'
if dataset == 'basset':
    filepath = os.path.join(data_path, 'basset_dataset.h5')
    data = load_basset_data(filepath, reverse_compliment=False)
elif dataset == 'deepsea':
    filepath = os.path.join(data_path, 'deepsea_dataset.h5')
    data = load_deepsea_data(filepath, reverse_compliment=False)
x_train, y_train, x_valid, y_valid, x_test, y_test = data
N, L, A = x_train.shape
num_labels = y_train.shape[1]

# build model
if model_name == 'CNN_ATT':
    model = models.CNN_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                           num_filters=num_filters, dense_units=1024, heads=16, key_size=128)

elif model_name == 'CNN_LSTM':
    model = models.CNN_LSTM(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                            num_filters=num_filters, lstm_units=256, dense_units=1024)

elif model_name == 'CNN_LSTM_ATT':
    model = models.CNN_LSTM_ATT(in_shape=(L,A), num_out=num_labels, activation=activation, pool_size=pool_size,
                                num_filters=num_filters, lstm_units=256, dense_units=1024, heads=16, key_size=128)

# compile model model
auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
model.compile(tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=[auroc, aupr])

# fit model
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_aupr', factor=0.2, patient=5, verbose=1, min_lr=1e-7, mode='max')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_aupr', patience=10, verbose=1, mode='max')
history = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_valid, y_valid), callbacks=[lr_decay, early_stop], verbose=1)

# save model params
model_dir = os.path.join(results_path, model_name+'_weights_'+str(trial)+'.h5')
model.save_weights(model_dir)

logs_dir = os.path.join(results_path, model_name+'_logs_'+str(trial)+'.pickle')
with open(logs_dir, 'wb') as handle:
    cPickle.dump(history.history, handle)

# Extract ppms from filters
ppms = utils.get_ppms(model, x_test)
logs_dir = os.path.join(results_path, model_name+'_filters_'+str(trial)+'.txt')
moana.meme_generate(ppms, output_file=motif_dir, prefix='filter')

# Tomtom analysis
tomtom_dir = os.path.join(results_path, model)
utils.tomtom(motif_dir, tomtom_dir)

# motif analysis
stats = utils.analysis(variant, motif_dir, tomtom_dir, model, x_test, y_test)
stats_dir = os.path.join(results_path, model_name+'_stats_'+str(trial)+'.npy')
np.save(stats_dir, stats, allow_pickle=True)


