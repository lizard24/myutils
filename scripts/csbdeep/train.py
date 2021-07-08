from __future__ import print_function, unicode_literals, absolute_import, division

from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

from myutils.general import copy_file

import matplotlib.pyplot as plt
import argparse
import numpy as np

import sys
import os



parser = argparse.ArgumentParser()

parser.add_argument("--file_npz", default=None, type=str, help="Npz file with training data.")

parser.add_argument("--model_name", default='my_model', type=str, help="Name of the folder the model is saved in.")
parser.add_argument("--model_folder", default='models' type=str, help="Folder the model is saved in.")

parser.add_argument("--val_split", default=0.05, type=float, help="Percentage of validation split.")

parser.add_argument("--train_loss", default="mae", type=str, help="Loss function")
parser.add_argument("--ms_ssim_no_weights", default=3, type=int, help="Number of weights if train_loss is ms_ssim.")
parser.add_argument("--ms_ssim_filter_size", default=11, type=int, help="Filter size of Gaussian window if train_loss is ms_ssim.")

parser.add_argument("--train_learning_rate", default=5e-3, type=float, help="Learning rate")

parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--train_steps_per_epoch", default=50, type=int, help="Steps per epoch")
parser.add_argument("--train_epochs", default=50, type=int, help="Epoch number")

parser.add_argument("--display", default=False, action='store_true')

parser.add_argument("--continue_training", default=False, action='store_true', help="Continue training with weights from last callback if existent.")
parser.add_argument("--callback_period", default=None, type=int, help='If/how often weights are saved in folder "callbacks".')

parser.add_argument("--model_name_pretrained", default=None, type=str, help="Folder to a pretrained model.")
parser.add_argument("--train_epochs_pretrained", default=None, type=int, help='Last epoch of model_name_pretrained.')

args = parser.parse_args()


if not os.path.exists('%s/%s/history.txt' % (args.model_folder, args.model_name) ):

    if not args.model_name_pretrained is None:
        args.continue_training = False
        args.train_epochs = args.train_epochs - args.train_epochs_pretrained
        print("Pre-trained model in '%s' is loaded. Training is continued at epoch %s." % (args.model_name_pretrained,
                                                                                           args.train_epochs_pretrained))
        for file_ in ['history.txt', 'weights_last.h5', 'weights_best.h5']:
            copy_file( source_file = file_,
                       source_dir  = args.model_name_pretrained,
                       target_dir  = '%s/%s' % (args.model_folder, args.model_name),
                       overwrite_target = True )
        for file_ in ['evaluation.png', 'evaluation2.png']:
            copy_file( source_file = file_,
                       target_file = 'pre-'+file_,
                       source_dir  = args.model_name_pretrained,
                       target_dir  = '%s/%s' % (args.model_folder, args.model_name),
                       overwrite_target = True )
            
    if args.continue_training:
        if os.path.exists('%s/%s/callbacks' % (args.model_folder, args.model_name)):
            callback_file = [item for item in os.listdir('%s/%s/callbacks' % (args.model_folder, args.model_name)) if 'hdf5' in item]
            if callback_file != []:
                for file_ in callback_file:
                    os.rename( '%s/%s/callbacks/%s'     % (args.folder_name, args.model_name, file_) ,
                               '%s/%s/callbacks/pre-%s' % (args.folder_name, args.model_name, file_) )
                callback_file = sorted(callback_file)[-1]
                args.train_epochs_pretrained = int(callback_file.split(".")[0].split('_')[-1])
                args.train_epochs            = args.train_epochs - args.train_epochs_pretrained
            else:
                args.continue_training = False
        else:
            args.continue_training = False

    if args.train_epochs<=0:
        print("'train_epochs<=0' - return without training.")
        return



(X,Y), (X_val,Y_val), axes = load_training_data( args.file_npz,
                                                 validation_split = args.val_split,
                                                 verbose = False )
    
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


config = Config( axes,
                 n_channel_in,
                 n_channel_out,
                 train_tensorboard     = False,
                 train_loss            = 'laplace'  if args.train_loss == 'probabilistic' else args.train_loss,
                 probabilistic         = True       if args.train_loss == 'probabilistic' else False,
                 ms_ssim_no_weights    = args.ms_ssim_no_weights,
                 ms_ssim_filter_size   = args.ms_ssim_filter_size,
                 train_learning_rate   = args.train_learning_rate,
                 train_batch_size      = args.train_batch_size,
                 train_steps_per_epoch = args.train_steps_per_epoch,
                 train_epochs          = args.train_epochs,
                 callback_period       = args.callback_period
               )


model = CARE(config, args.model_name, basedir=args.model_folder)

if args.display:
    model.keras_model.summary()

if not args.model_name_pretrained is None:
    model.load_weights('weights_best.h5')
    print("Weights are loaded from pre-trained model.")
elif continue_training:
    model.load_weights('callbacks/%s' % callback_file)
    print("Weights are loaded from earlier callback: %s" % callback_file)
    

history = model.train(X,Y, validation_data=(X_val,Y_val))


cols = list(history.history.keys())    
        
fig, ax = plt.figure(figsize=(16,5))
ax.plot_history(history,[cols[4],cols[0]],[cols[5],cols[1],cols[6],cols[2], cols[7],cols[3]]);
fig.savefig( "%s/%s/loss.png" % (args.model_folder, args.model_name) )
if args.display:
    plt.show(fig)
plt.close(fig)

fig, ax = plt.figure(figsize=(20,12))
_P = model.keras_model.predict(X_val[:5])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
ax.plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
fig.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');
fig.savefig( "%s/%s/evaluation.png" % (args.model_folder, args.model_name) )
if args.display:
    plt.show(fig)
plt.close(fig)

if (args.continue_training is False) and (args.model_name_pretrained is None):
    history_dict = {'epoch no': list(range(1,args.train_epochs+1)),
                    **history.history}
else:
    history_dict = {'epoch no': list(range(args.train_epochs_pretrained+1,args.train_epochs+args.train_epochs_pretrained+1)),
                    **history.history}

cols.insert(0,'epoch no')

if not os.path.exists('%s/%s/history.txt' % (args.model_folder, args.model_name)):
    file = open("%s/%s/history.txt" % (args.model_folder, args.model_name),"a")
    file.write("\t".join(cols))
    file.write("\n")
    file.close()

file = open("%s/%s/history.txt" % (args.model_folder, args.model_name),"a")    
for i in range(len(history_dict[cols[0]])):
    for j in range(len(cols)):
        file.write(str(history_dict[cols[j]][i])+"\t")
    file.write("\n")
file.close()