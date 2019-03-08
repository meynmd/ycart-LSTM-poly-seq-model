from dataset import Dataset, ground_truth, safe_mkdir
from model import Model, make_model_from_dataset
from display_utils import display_prediction, load_piano_roll, load_quant_piano_roll

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys


crop = True
quant = True


note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]
n_notes = note_max - note_min + 1
fs=4


n_hiddens = [128,256]
learning_rates = [0.001,0.01]
max_len = 60 #1 minute files only


midi_file = 'data/synth/test/D4F4A4_B3D4_D3_012.mid'
piano_roll = load_quant_piano_roll(midi_file,note_range,fs)
n_steps = piano_roll.shape[1]

base_path = "piano-midi/quant_augm/"

for n_hidden in n_hiddens:
    for learning_rate in learning_rates:
        model = Model(n_notes=n_notes,n_steps=n_steps,n_hidden=n_hidden,learning_rate=learning_rate)
        save_path = base_path+str(n_hidden)+'_'+str(learning_rate)
        display_prediction(midi_file,
            model,save_path,note_range,fs=fs,sigmoid=True,save=False)
