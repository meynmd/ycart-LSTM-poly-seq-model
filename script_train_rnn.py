from dataset import Dataset
from model import Model, make_model_from_dataset

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys


# Restrict the range of notes to [note_min, note_max] instead of [0,128]
crop = True
#Note based timesteps ?
quant = True
#Data augmentation ?
augm = True

note_range = [21,108]
note_min = note_range[0]
note_max = note_range[1]
fs=4

n_hiddens = [128,256] #number of features in hidden layer
learning_rates = [0.001,0.01]
epochs = 50
batch_size = 50
display_per_epoch = 5
save_step = 1
max_len = 60 #60 seconds files only


print "Computation start : "+str(datetime.now())

data_path = 'data/piano-midi/'

data = Dataset()
if quant :
    data.load_data_quant(data_path,note_min=note_min,note_max=note_max,
        fs=fs,crop=crop,max_len=max_len)
else :
    data.load_data(data_path,note_min=note_min,note_max=note_max,
        fs=fs,crop=crop,max_len=max_len)
if augm :
    data.transpose_all()


base_path = "piano-midi/quant_augm/"

for n_hidden in n_hiddens:
    for learning_rate in learning_rates:

        save_path = base_path+str(n_hidden)+"_"+str(learning_rate)+"/"

        print "________________________________________"
        print "Hidden nodes = "+str(n_hidden)+", Learning rate = "+str(learning_rate)
        print "________________________________________"
        print "."

        model = make_model_from_dataset(data,n_hidden=n_hidden,learning_rate=learning_rate)
        model.train(data,save_path=save_path,
            epochs=epochs,batch_size=batch_size,display_per_epoch=display_per_epoch,
            save_step=save_step,summarize=True)
        tf.reset_default_graph()
        print "."
        print "."
        print "."

print "Computation end : "+str(datetime.now())
