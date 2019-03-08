==================
PRESENTATION
==================

In this archive can be found all the data and code necessary to reproduce the
experiments described in :
Adrien Ycart and Emmanouil Benetos.
“A Study on LSTM Networks for Polyphonic Music Sequence Modelling”
18th International Society for Music Information Retrieval Conference (ISMIR),
October 2017, Suzhou, China.

==================
CODE ORGANISATION
==================

The datasets can be found in the 'data/' folder, split in train, valid and test
subsets.

dataset.py holds functions relative to the loading and manipulation of
the dataset

model.py holds functions defining the model, as well as the training
process.

eval_utils.py holds function to compute the evaluation metrics of a given model.

display_utils.py holds functions to display some MIDI files and some predictions.

script_train_rnn.py is a script allowing to train several models.

script_get_measures.py is a script allowing, once some model are trained,
to compute the best threshold on the validation dataset, and use that threshold
to compute the evaluation metrics on the test dataset.

script_display_prediction.py is a script allowing, once some model are trained,
to visualise the predictions made by these models with a given MIDI file.

==================
DATA SAVING
==================

For most training and loading functions, you are asked a "base_path".
This variable allows to specify a base path, characterising the current experiment.

Inside that folder, some subfolders will be created for each
(n_hidden,learning_rate) configuration.

When running the training script, the checkpoint files are saved in the folder
'ckpt/', and the summary files are saved in 'summ/'

EXAMPLE :
I run an experiment comparing models with n_hiddens = [128, 256] and
learning_rates = [0.001, 0.01], on the Synth dataset, with quantised timesteps.
I choose as a base_path : 'synth/quantised/'.
The checkpoint files for the model (128,0.01) will be found in :
'ckpt/synth/quantised/128_0.01/'
The summary files for this same model will be found in :
'summ/synth/quantised/128_0.01/'

==================
KNOWN ISSUES
==================

This was some preliminary code, there were some mistakes, that were kept
for the sake of reproducibility.
The user is free to fix them, if they are willing to.

Among the known issues are :
 - The dataset is not shuffled between epochs.
 - When computed with Tensorflow and not Numpy, the evaluation metrics sometimes
 output NaN : this is due to the fact that the denominator is negative, and can
 easily be fixed by adding a small float value to the denominator.
 - The function _run_by_batch only works with cross_entropy, not with prediction.
 - The length of each sequence is not provided as argument to the dynamic_rnn.
 - When performing data augmentation, all datasets are augmented, while only
 the training dataset should be.

If there were any other issues, please contact the author (Adrien Ycart) at :
a.ycart@qmul.ac.uk
