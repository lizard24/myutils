from __future__ import print_function, unicode_literals, absolute_import, division

from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

from myutils.general   import copy_file
from myutils.mycsbdeep import get_history

import matplotlib.pyplot as plt
import argparse
import numpy as np

import os



parser = argparse.ArgumentParser()

parser.add_argument("--file_npz", default='traig-data.npz', type=str, help="Npz file with training data.")

parser.add_argument("--model_name", default='my_model', type=str, help="Name of the folder the model is saved in.")
parser.add_argument("--model_folder", default='models', type=str, help="Folder the model is saved in.")

parser.add_argument("--val_split", default=0.05, type=float, help="Percentage of validation split.")

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

if os.path.exists('%s/%s/history.txt' % (args.model_folder, args.model_name) ):
    print("FYI: Folder already exists: '%s/%s'!" % (args.model_folder, args.model_name))


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

        
print(args.continue_training)

if not (args.model_name_pretrained is None) or (args.continue_training is True):
    history_pre = get_history('%s/%s/history.txt' % (args.model_folder, args.model_name))
    print("Learning rate is modified: %s -> %s !" % (args.train_learning_rate, history_pre['lr'][-1]) )
    args.train_learning_rate = history_pre['lr'][-1]
else:
    history_pre = None


if args.train_epochs<=0:
    print("'train_epochs<=0' - return without training.")
    sys.exit()

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
                 callback_period       = args.callback_period,
                 callback_start        = args.train_epochs_pretrained
               )


model = CARE(config, args.model_name, basedir=args.model_folder)

if args.display:
    model.keras_model.summary()

if not args.model_name_pretrained is None:
    model.load_weights('weights_best.h5')
    print("Weights are loaded from pre-trained model.")
elif args.continue_training:
    model.load_weights('callbacks/%s' % callback_file)
    print("Weights are loaded from earlier callback: %s" % callback_file)

history = model.train(X, Y, validation_data = (X_val, Y_val))


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


cols =  ['epoch_no', 'loss', 'val_loss', 'mae', 'val_mae', 'mse', 'val_mse', 'ssim', 'val_ssim', 'lr']

if not history_pre is None:
    for col in cols:
        history.history[col] = history_pre[col] + history.history[col] 

        
plt.figure(figsize=(16,5))
#cols = list(history.history.keys())
plot_history(history,['loss', 'val_loss'],['mse', 'val_mse', 'mae', 'val_mae', 'ssim', 'val_ssim'])
plt.savefig( "%s/%s/loss.png" % (args.model_folder,
                                 args.model_name) )
if args.display:
    plt.show()
plt.close()

if not os.path.exists('%s/%s/history.txt' % (args.model_folder, args.model_name)):
    file = open("%s/%s/history.txt" % (args.model_folder, args.model_name),"a")
    file.write("%s\n" % "\t".join(cols))
else:
    file = open("%s/%s/history.txt" % (args.model_folder, args.model_name),"a")
    
for row in range(len(history.history[cols[0]])):
    for col in cols:
        file.write(str(history.history[col][row])+"\t")
    file.write("\n")

file.close()
