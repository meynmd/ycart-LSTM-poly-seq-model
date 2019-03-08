import os
import numpy as np
import pretty_midi as pm
from random import shuffle
import cPickle as pickle
from datetime import datetime
import copy

class Dataset:
    """Classe representing the dataset."""

    def __init__(self):
        self.train = []
        self.test = []
        self.valid = []

        self.note_min = 0
        self.note_max = 127



    def load_data_one(self,folder,subset,note_min=0,note_max=127,fs=100,max_len=None,crop=True):
        dataset = []
        subfolder = os.path.join(folder,subset)
        for fn in os.listdir(subfolder):
            if fn.endswith('.mid') and not fn.startswith('.'):
                midi_data = pm.PrettyMIDI(os.path.join(subfolder,fn))
                piano_roll = midi_data.get_piano_roll(fs=fs)
                dataset += [piano_roll]
        setattr(self,subset,dataset)


    def load_data(self,folder,note_min=0,note_max=127,fs=100,max_len=None,crop=True):
        #Loads the dataset in folder, containing subfolders train, valid and test.
        #Uses time-based timesteps.
        #fs is given in Hz.
        #max_len is given in seconds


        self.note_min = note_min
        self.note_max = note_max
        for subset in ["train","valid","test"]:
            self.load_data_one(folder,subset,note_min,note_max,fs,crop)
            self.__binarize_one(subset)
            if crop :
                self.__crop_one(subset)
        self.__zero_pad(max_len,fs=fs)
        print "Dataset loaded ! "+str(datetime.now())

    def load_data_quant_one(self,folder,subset,note_min=0,note_max=127,fs=4,max_len=None,crop=True):
        dataset = []
        subfolder = os.path.join(folder,subset)
        for fn in os.listdir(subfolder):
            if fn.endswith('.mid') and not fn.startswith('.'):
                midi_data = pm.PrettyMIDI(os.path.join(subfolder,fn))
                piano_roll = get_quant_piano_roll(midi_data,fs,max_len)
                dataset += [piano_roll]
        setattr(self,subset,dataset)

    def load_data_quant(self,folder,note_min=0,note_max=127,fs=4,max_len=None,crop=True):
        #Loads the dataset in folder, containing subfolders train, valid and test.
        #Uses note-based timesteps.
        #fs is the number of frames per beat.
        #max_len is given in seconds

        self.note_min = note_min
        self.note_max = note_max
        for subset in ["train","valid","test"]:
            self.load_data_quant_one(folder,subset,note_min,note_max,fs,max_len,crop)
            self.__binarize_one(subset)
            if crop :
                self.__crop_one(subset)
        self.__zero_pad()
        print "Dataset loaded ! "+str(datetime.now())

    def get_n_files(self,subset):
        return getattr(self,subset).shape[0]
    def get_len_files(self):
        return self.train.shape[2]
    def get_n_notes(self):
        return self.train.shape[1]
    def get_note_range(self):
        return [self.note_min,self.note_max]

    def __binarize_one(self,subset):
        data = getattr(self,subset)
        binarized = []
        for datum in data:
            binarized += [np.not_equal(datum,np.zeros(datum.shape))]#.astype(int)
        setattr(self,subset,binarized)

    def __max_len(self,dataset):
        if dataset == []:
            return 0
        else :
            return max(map(lambda x: x.shape[1], dataset))

    def __zero_pad(self,max_len=None,fs=None):
        if max_len == None:
            max_train = self.__max_len(self.train)
            max_valid = self.__max_len(self.valid)
            max_test = self.__max_len(self.test)
            max_len = max([max_train,max_valid,max_test])
        else:
            max_len=max_len*fs
        for subset in ["train","valid","test"]:
            self.__zero_pad_one(subset,max_len)


    def __zero_pad_one(self,subset,max_len):
        #Zero-padding the dataset
        dataset = getattr(self,subset)
        N = len(dataset)
        N_notes = len(dataset[0])
        dataset_padded = np.zeros([N,N_notes,max_len])
        for i in range(N):
            data = dataset[i]
            if data.shape[1] >= max_len:
                dataset_padded[i] = data[:,0:max_len]
            else :
                dataset_padded[i] = np.pad(data,pad_width=((0,0),(0,max_len-data.shape[1])),mode='constant')
        setattr(self,subset,dataset_padded)

    def __crop_one(self,subset):
        data = getattr(self,subset)
        cropped_subset = []
        for datum in data:
            cropped_subset += [datum[self.note_min:self.note_max+1,:]]
        setattr(self,subset,cropped_subset)

    def __transpose(self,piano_roll,i):
        if i==0:
            return piano_roll
        elif i>0:
            return np.append(piano_roll[i:,:],np.zeros([i,piano_roll.shape[1]]),0)
        elif i<0:
            return np.append(np.zeros([-i,piano_roll.shape[1]]),piano_roll[:i,:],0)

    def transpose_all_one(self,subset):
        data = getattr(self,subset)
        tr_range = [-5,7]
        new_data = np.zeros([data.shape[0]*12,data.shape[1],data.shape[2]])
        i=0
        for datum in data:
            for j in range(*tr_range):
                tr_datum = self.__transpose(datum,j)
                print j
                print tr_datum
                new_data[i] = tr_datum
                i+=1
        setattr(self,subset,new_data)

    def transpose_all(self):
        #Transposes the datasets in all keys between a fouth below and a fifth above
        for subset in ["train","valid","test"]:
            self.transpose_all_one(subset)


def get_quant_piano_roll(midi_data,fs=4,max_len=None):
    #Returns from a PrettyMIDI object the piano-roll with note-based timesteps
    data = copy.deepcopy(midi_data)

    PPQ = float(data.resolution)

    for instr in data.instruments:
        for note in instr.notes:
            note.start = data.time_to_tick(note.start)/PPQ
            note.end = data.time_to_tick(note.end)/PPQ


    quant_piano_roll = data.get_piano_roll(fs)
    if not max_len == None:
        end = int(round(data.time_to_tick(max_len)/PPQ*fs))
        quant_piano_roll = quant_piano_roll[:,0:end]

    return quant_piano_roll


def ground_truth(data):
    return data[:,:,1:]

def move_files(file_list,folder1,folder2):
    for midi_file in file_list:
        os.rename(os.path.join(folder1,midi_file), os.path.join(folder2,midi_file))

def safe_mkdir(dir,clean=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if clean and not os.listdir(dir) == [] :
        old_path = os.path.join(dir,"old")
        safe_mkdir(old_path)
        for fn in os.listdir(dir):
            full_path = os.path.join(dir,fn)
            if not os.path.isdir(full_path):
                os.rename(full_path,os.path.join(old_path,fn))




def split_files(folder,test=0.2,valid=0.1):

    midi_list = [x for x in os.listdir(folder) if x.endswith('.mid')]

    train_path = os.path.join(folder,"train/")
    valid_path = os.path.join(folder,"valid/")
    test_path = os.path.join(folder,"test/")

    safe_mkdir(train_path)
    safe_mkdir(valid_path)
    safe_mkdir(test_path)

    N = len(midi_list)
    N_test = int(N*test)
    N_valid = int(N*valid)

    shuffle(midi_list)
    test_list, valid_list, train_list = midi_list[:N_test], midi_list[N_test:N_test+N_valid],midi_list[N_test+N_valid:]

    move_files(test_list,folder,test_path)
    move_files(valid_list,folder,valid_path)
    move_files(train_list,folder,train_path)

def unsplit_files(folder):
    train_path = os.path.join(folder,"train/")
    valid_path = os.path.join(folder,"valid/")
    test_path = os.path.join(folder,"test/")

    move_files(os.listdir(train_path),train_path,folder)
    move_files(os.listdir(valid_path),valid_path,folder)
    move_files(os.listdir(test_path),test_path,folder)

    os.rmdir(train_path)
    os.rmdir(test_path)
    os.rmdir(valid_path)
