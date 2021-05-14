# Imports
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from tfomics import moana, evaluate
from tfomics.layers import MultiHeadAttention

import numpy as np
import requests as rq
import io, h5py, os
from six.moves import cPickle

import models
import utils

# Retrieve dataset
x_train, y_train, x_valid, y_valid, x_test, y_test = utils.get_synthetic_dataset()

def execute_pipeline(baseline, category, variant, trial, model, epochs):
    
    global x_train, y_train, x_valid, y_valid, x_test, y_test

    # Create directories
    model_dir = f'{baseline}/models/{category}/model-{variant}'
    motif_dir = f'{baseline}/motifs/{category}/model-{variant}'
    tomtom_dir = f'{baseline}/tomtom/{category}/model-{variant}'
    stats_dir = f'{baseline}/stats/{category}/model-{variant}'
    logs_dir = f'{baseline}/history/{category}/model-{variant}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(motif_dir):
        os.makedirs(motif_dir)
    if not os.path.exists(tomtom_dir):
        os.makedirs(tomtom_dir)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    model_dir += f'/trial-{trial}/weights'
    motif_dir += f'/trial-{trial}.txt'
    tomtom_dir += f'/trial-{trial}'
    stats_dir += f'/trial-{trial}.npy'
    logs_dir += f'/trial-{trial}.pickle'


    # Train model
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    model.compile(tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=[auroc, aupr])

    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_aupr', factor=0.2, patient=5, verbose=1, min_lr=1e-7, mode='max')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_aupr', patience=15, verbose=1, mode='max')
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid), callbacks=[lr_decay, early_stop], verbose=1)

    model.save_weights(model_dir)
    
    with open(logs_dir, 'wb') as handle:
        cPickle.dump(history.history, handle)


    # Extract ppms from filters
    ppms = utils.get_ppms(model, x_test)
    moana.meme_generate(ppms, output_file=motif_dir, prefix='filter')


    # Tomtom analysis
    utils.tomtom(motif_dir, tomtom_dir)


    # Analysis
    stats = utils.analysis(variant, motif_dir, tomtom_dir, model, x_test, y_test)
    np.save(stats_dir, stats, allow_pickle=True)

    