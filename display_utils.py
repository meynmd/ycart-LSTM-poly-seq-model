from dataset import Dataset, ground_truth, safe_mkdir, get_quant_piano_roll
from model import Model, make_model_from_dataset

import os

import tensorflow as tf
import pretty_midi as pm
import matplotlib.pyplot as plt
import numpy as np

def print_measures(F,pre,rec):
    print "F : "+str(F)+", pre : "+str(pre)+", rec : "+str(rec)


def load_piano_roll(midi_file,crop,fs=100):
    midi = pm.PrettyMIDI(midi_file)
    piano_roll = midi.get_piano_roll(fs)
    #Binarize and crop the piano_roll
    piano_roll = np.not_equal(piano_roll,0).astype(int)
    if crop:
        piano_roll = piano_roll[crop[0]:crop[1]+1,:]
    return piano_roll

def load_quant_piano_roll(midi_file,crop,fs=4,max_len=None):
    midi = pm.PrettyMIDI(midi_file)
    piano_roll = get_quant_piano_roll(midi,fs,max_len)
    #Binarize and crop the piano_roll
    piano_roll = np.not_equal(piano_roll,0).astype(int)
    if crop:
        piano_roll = piano_roll[crop[0]:crop[1]+1,:]
    return piano_roll



def compare_piano_rolls(piano_roll1,piano_roll2,crop,title=""):
    if crop:
        labels = list(range(crop[0],crop[1]+1))
    else :
        labels = list(range(0,128))
    labels = [pm.note_number_to_name(x) for x in labels]

    plt.figure()

    plt.subplot(211)
    plt.imshow(piano_roll1,aspect='auto',origin='lower')
    plt.yticks([x+0.5 for x in list(range(len(labels)))] , labels,fontsize=5)
    ax = plt.gca()
    ax.grid(True,axis='y',color='black')
    plt.title(title)
    plt.subplot(212)
    plt.imshow(piano_roll2,aspect='auto',origin='lower')
    plt.yticks([x+0.5 for x in list(range(len(labels)))] , labels,fontsize=5)
    ax = plt.gca()
    ax.grid(True,axis='y',color='black')


def display_prediction(midi_file,model,save_path,crop,fs=100,n_model=None,sigmoid=False,save=False,quant=False):
    if quant :
        piano_roll = load_quant_piano_roll(midi_file,crop,fs)
    else :
        piano_roll = load_piano_roll(midi_file,crop,fs)
    piano_roll = np.asarray([piano_roll])
    # print piano_roll.shape
    # print np.transpose(piano_roll,[0,2,1]).shape
    pred = model.run_prediction(piano_roll,save_path,n_model,sigmoid)
    pred = pred[0]
    target = ground_truth(piano_roll)
    target = target[0]

    midi_name = os.path.splitext(os.path.basename(midi_file))[0]
    title = os.path.basename(save_path)+", "+midi_name

    compare_piano_rolls(target,pred,crop,title)

    if save :
        if sigmoid:
            fig_save_path = os.path.join("./fig/",save_path,midi_name+'_sigmoid.png')
        else:
            fig_save_path = os.path.join("./fig/",save_path,midi_name+'.png')
        safe_mkdir(fig_save_path)
        plt.savefig(fig_save_path)
    else :
        plt.show()
