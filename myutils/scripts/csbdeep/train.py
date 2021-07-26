from __future__ import print_function, unicode_literals, absolute_import, division

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import sys
import os

from tifffile import imread
from six      import string_types

from csbdeep.utils  import axes_dict, plot_some, plot_history
from csbdeep.io     import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.data   import shuffle_inplace

from myutils.general   import copy_file
from myutils.mycsbdeep import get_history



########## FUNCTIONS

def prepare_data(x,y):
    shuffle_inplace(x,y)
    x, y = np.asarray(x), np.asarray(y)
    if len(x.shape) != len(axes):
        cidx = axes.find('C')
        cidx = axes.find('C')
        x, y = map(lambda z: np.reshape(z, z.shape[:cidx]+(1,)+z.shape[cidx:]), (x, y))
    return x, y

def plot_history_v2(history,*keys,**kwargs):

    logy = kwargs.pop('logy',False)

    if all(( isinstance(k,string_types) for k in keys )):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    print(history.keys())
    
    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1,w,i+1)
        for k in ([group] if isinstance(group,string_types) else group):
            plt.plot(history['epoch_no'],history[k],'.-',label=k,**kwargs)
            if logy:
                plt.gca().set_yscale('log', nonposy='clip')
        plt.xlabel('epoch')
        plt.legend(loc='best')
        
   
        
########## ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument("--file_npz", default=None, type=str, help="Npz file with training data.")

parser.add_argument("--folder_data", default=None, type=str, help="Alternative to 'file_npz'. 'folder_data' is directory to subfolders that contain training data ('input', target', 'target_val', 'input_val').")
parser.add_argument("--axes", default='SYXC', type=str, help="Axes of training data. Needs to be defined if 'file_npz' is None.")

parser.add_argument("--model_name", default='my_model', type=str, help="Name of the folder the model is saved in.")
parser.add_argument("--model_folder", default='models', type=str, help="Folder the model is saved in.")

parser.add_argument("--val_split", default=0.1, type=float, help="Percentage of validation split.")

parser.add_argument("--train_loss", default="mae", type=str, help="Loss function")
parser.add_argument("--ms_ssim_no_weights", default=3, type=int, help="Number of weights if train_loss is ms_ssim.")
parser.add_argument("--ms_ssim_filter_size", default=11, type=int, help="Filter size of Gaussian window if train_loss is ms_ssim.")

parser.add_argument("--train_learning_rate", default=5e-3, type=float, help="Learning rate")

parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--train_steps_per_epoch", default=50, type=int, help="Steps per epoch")
parser.add_argument("--train_epochs", default=50, type=int, help="Epoch number")

parser.add_argument("--display", action='store_true')

parser.add_argument("--continue_training", action='store_true', help="Continue training with weights from last callback if existent. Default: False")
parser.add_argument("--callback_period", default=None, type=int, help='If/how often weights are saved in folder "callbacks".')

parser.add_argument("--model_name_pretrained", default=None, type=str, help="Folder to a pretrained model.")
parser.add_argument("--train_epochs_pretrained", default=0, type=int, help='Last epoch of model_name_pretrained.')

args, unknown = parser.parse_known_args()



########## LOAD PREVIOUS MODELS

if os.path.exists('%s/%s/history.txt' % (args.model_folder, args.model_name) ):
    print("Note: Folder already exists: '%s/%s'!" % (args.model_folder, args.model_name))

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

if args.continue_training:
    if os.path.exists('%s/%s/callbacks' % (args.model_folder, args.model_name)):
        callback_file = [item for item in os.listdir('%s/%s/callbacks' % (args.model_folder, args.model_name)) if 'hdf5' in item]
        if callback_file != []:
            callback_file = sorted(callback_file)[-1]
            args.train_epochs_pretrained = int(callback_file.split(".")[0].split('_')[-1])
            args.train_epochs            = args.train_epochs - args.train_epochs_pretrained
        else:
            args.continue_training = False
    else:
        args.continue_training = False

if os.path.exists('%s/%s/history.txt' % (args.model_folder, args.model_name)):
    if not (args.model_name_pretrained is None) or (args.continue_training is True):
        history_all = get_history('%s/%s/history.txt' % (args.model_folder, args.model_name))
        if args.train_learning_rate != history_all['lr'][-1]:
            print("Learning rate is modified: %s -> %s !" % (args.train_learning_rate, history_all['lr'][-1]) )
            args.train_learning_rate = history_all['lr'][-1]
else:
    history_all = {'epoch_no': [n+1 for n in range(args.train_epochs)]}

if args.train_epochs<=0:
    print("'train_epochs' <= 0 - return without training.")
    sys.exit()



    
########## GET TRAINING DATA
    
if not args.file_npz is None:
    
    (X,Y), (X_val,Y_val), axes = load_training_data( args.file_npz,
                                                     validation_split = args.val_split,
                                                     verbose = False )
    
elif not (args.folder_data is None) and (os.path.isdir(args.folder_data)):
    
    X     = [ imread('/'.join([args.folder_data, 'input'     , file])) for file in os.listdir('%s/input'      % args.folder_data) ]
    Y     = [ imread('/'.join([args.folder_data, 'target'    , file])) for file in os.listdir('%s/target'     % args.folder_data) ]
    X_val = [ imread('/'.join([args.folder_data, 'input_val' , file])) for file in os.listdir('%s/input_val'  % args.folder_data) ]
    Y_val = [ imread('/'.join([args.folder_data, 'target_val', file])) for file in os.listdir('%s/target_val' % args.folder_data) ]

    axes = args.axes
        
    X, Y         = prepare_data(X, Y)
    X_val, Y_val = prepare_data(X_val, Y_val)
    
else:
    print("Define 'file_npz' or 'folder_data' - return without execution.")
    sys.exit()
    



########## START TRAINING

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


config = Config( axes,
                 n_channel_in,
                 n_channel_out,
                 train_tensorboard     = True       if args.display else False,
                 train_loss            = 'laplace'  if args.train_loss == 'probabilistic' else args.train_loss,
                 probabilistic         = True       if args.train_loss == 'probabilistic' else False,
                 ms_ssim_no_weights    = args.ms_ssim_no_weights,
                 ms_ssim_filter_size   = args.ms_ssim_filter_size,
                 train_learning_rate   = args.train_learning_rate,
                 train_batch_size      = args.train_batch_size,
                 train_steps_per_epoch = args.train_steps_per_epoch,
                 train_epochs          = args.train_epochs,
                 callback_period       = args.callback_period,
                 callback_start        = args.train_epochs_pretrained )


model = CARE(config, args.model_name, basedir=args.model_folder)

if args.display:
    model.keras_model.summary()

if not args.model_name_pretrained is None:
    model.load_weights('weights_best.h5')
    print("Weights are loaded from pre-trained model.")
elif args.continue_training:
    model.load_weights('callbacks/%s' % callback_file)
    print("Weights are loaded from earlier callback: %s" % callback_file)

history = model.train( X, Y, validation_data = (X_val, Y_val),
                       verbose = 1 if args.display else 0 )  ### 0 = silent, 1 = progress bar, 2 = one line per epoch




########## EXPORT: FIGURES & HISTORY

plt.figure(figsize=(20,12))
_P = model.keras_model.predict(X_val[:5])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');
plt.savefig( "%s/%s/evaluation.png" % (args.model_folder, args.model_name) )
if args.display:
    plt.show()
plt.close()

for col in history.history.keys():
    history_all[col] = history_all[col] + history.history[col] if col in history_all.keys() else history.history[col]

plt.figure(figsize=(16,5))
plot_history_v2(history_all,['loss', 'val_loss'],['mse', 'val_mse', 'mae', 'val_mae', 'ssim', 'val_ssim'])
plt.savefig( "%s/%s/loss.png" % (args.model_folder, args.model_name) )
if args.display:
    plt.show()
plt.close()


cols =  ['epoch_no', 'loss', 'val_loss', 'mae', 'val_mae', 'mse', 'val_mse', 'ssim', 'val_ssim', 'lr']

file = open("%s/%s/history.txt" % (args.model_folder, args.model_name), "w") ### overwrites history.txt
file.write("%s\n" % "\t".join(cols))

for row in range(len(history_all[cols[0]])):
    file.write( "%s\n" % "\t".join( [str(history_all[col][row]) for col in cols] ) )
    
file.close()