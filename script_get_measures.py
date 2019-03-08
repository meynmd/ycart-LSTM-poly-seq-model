from dataset import Dataset, ground_truth, safe_mkdir
from model import Model, make_model_from_dataset
from eval_utils import get_best_eval_metrics

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys


crop = True
quant = True
save_results = False

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]
fs=4


n_hiddens = [128,256]
learning_rates = [0.001,0.01]
max_len = 60 #1 minute files only

print "Computation start : "+str(datetime.now())

data = Dataset()

data_path = 'data/piano-midi/'

data = Dataset()
if quant :
    data.load_data_quant(data_path,note_min=note_min,note_max=note_max,
        fs=fs,crop=crop,max_len=max_len)
else :
    data.load_data(data_path,note_min=note_min,note_max=note_max,
        fs=fs,crop=crop,max_len=max_len)


if save_results:
    #Store all the results in a 2d dictionnary
    results = {}

base_path = "piano-midi/quant_augm/"

for n_hidden in n_hiddens:
    results[n_hidden]={}
    for learning_rate in learning_rates:
        save_path = base_path+str(n_hidden)+"_"+str(learning_rate)+"/"


        model = make_model_from_dataset(data,n_hidden=n_hidden,learning_rate=learning_rate)

        F, prec, rec = get_best_eval_metrics(data,model,save_path,verbose=True)

        results[n_hidden][learning_rate] = [F, prec, rec]

        tf.reset_default_graph()

if save_results:
    import cPickle as pickle
    pickle.dump(results, open(os.path.join("ckpt",base_path,'results.p'), "wb"))

print "Computation end : "+str(datetime.now())
