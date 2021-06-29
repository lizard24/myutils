from __future__ import print_function, unicode_literals, absolute_import, division

from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

import matplotlib.pyplot as plt
import argparse
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--file_npz", default=None, type=str, help="Npz file with training data.")
parser.add_argument("--model_name", default='my_model', type=str, help="Name of the folder the model is saved in.")

parser.add_argument("--val_split", default=0.05, type=float, help="Percentage of validation split.")

parser.add_argument("--train_loss", default="mae", type=str, help="Loss function")
parser.add_argument("--ms_ssim_no_weights", default=3, type=int, help="Number of weights if train_loss is ms_ssim.")
parser.add_argument("--ms_ssim_filter_size", default=11, type=int, help="Filter size of Gaussian window if train_loss is ms_ssim.")

parser.add_argument("--train_learning_rate", default=5e-3, type=float, help="Learning rate")

parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--train_steps_per_epoch", default=50, type=int, help="Steps per epoch")
parser.add_argument("--train_epochs", default=50, type=int, help="Epoch number")

args = parser.parse_args()


(X,Y), (X_val,Y_val), axes = load_training_data( args.file_npz,
                                                 validation_split = args.val_split,
                                                 verbose = False )

plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)');

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


config = Config( axes,
                 n_channel_in,
                 n_channel_out,
                 unet_kern_size = 3,
                 train_loss            = args.train_loss,
                 ms_ssim_no_weights    = args.ms_ssim_no_weights,
                 ms_ssim_filter_size   = args.ms_ssim_filter_size,
                 train_learning_rate   = args.train_learning_rate,
                 train_batch_size      = args.train_batch_size,
                 train_steps_per_epoch = args.train_steps_per_epoch,
                 train_epochs          = args.train_epochs )


model = CARE(config, args.model_name, basedir='models')

#model.keras_model.summary()

history = model.train(X,Y, validation_data=(X_val,Y_val))

plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
plt.savefig("models/%s/loss.png" % args.model_name)
plt.close()

plt.figure(figsize=(20,12))
_P = model.keras_model.predict(X_val[:5])
plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');
plt.savefig("models/%s/evaluation.png" % args.model_name)
plt.close()

