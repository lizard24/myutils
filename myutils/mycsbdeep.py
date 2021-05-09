import numpy as np
import matplotlib.pyplot as plt

from csbdeep.utils    import axes_dict, plot_some, plot_history, Path
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.models   import Config, CARE

import os

from myutils.general        import generate_folder, copy_file
from myutils.myimageprocess import im



def change_configfile(directory):

    os.rename("%s/config.json" % directory, "%s/config_v0.json" % directory)

    with open("%s/config_v0.json" % directory, "r") as fin:
        for item in fin:
            item = item.replace('ms_ssim_no_weights', 'no_weights').replace('no_weights', 'ms_ssim_no_weights')
            item = item.replace('ms_ssim_filter_size', 'filter_size').replace('filter_size', 'ms_ssim_filter_size')
            item = item.replace('mssim', 'ms_ssim') 
            with open("%s/config.json" % directory, "wt") as fout:
                fout.write(item)


def training(  X, Y, X_val, Y_val, axes,
               basedir               = None,
               unet_n_depth          = 3,
               train_batch_size      = 8,
               train_steps_per_epoch = 50,
               train_epochs          = 50,
               train_learning_rate   = 0.004,
               train_loss            = 'mae',
               ms_ssim_no_weights    = 3,
               ms_ssim_filter_size   = 11,
               continue_training     = False,
               model_name_pretrained = None,
               callback_period       = None ):
    
    """Conducts training of a CARE network for specified training data.

    Parameters
    ----------
    X, Y, X_val, Y_val    : numpy arrays with axes 'STYXC' or 'SYXC'
    axes                  : str
    unet_n_depth          : int
    train_batch_size      : int
    train_steps_per_epoch : int
    train_epochs          : int
    train_learning_rate   : float
    train_loss            : string; 'mae', 'mse', 'ssim' or 'ms_ssim'
    ms_ssim_no_weights    : int or 'None'; specify weights for ms_ssim between 1-5
                            default: 3 if train_loss is set to 'ms_ssim'
    ms_ssim_filter_size   : int or 'None'; specify filter_size for ms_ssim
                            requirement: filter_size >= patch_size/2^(weights-1)
                            default: 11 if train_loss is set to 'ms_ssim'
    continue_training     : boolean; continue training with weights from last callback if existent
    model_name_pretrained : str; folder to a pretrained model; takes precedence over 'continue_training'
    callback_period       : int or 'None'; if/how often are weights saved in 'callbacks' folder
    
    Return
    ------
    Trained model in basedir.
    
    """
    
    model_name = '%s-%s-%s_%s_%s' % ( train_epochs,
                                      train_steps_per_epoch,
                                      train_batch_size,
                                      'ms%s' % ms_ssim_no_weights if train_loss == 'ms_ssim' else train_loss,
                                      "{:.0e}".format(train_learning_rate).replace('0','') )
    
    folder_out = '%s/%s' % (basedir, model_name)

    if not os.path.exists('%s/history.txt' % folder_out):
    
        if not model_name_pretrained is None:
            continue_training = False
            train_epochs_pretrained = int(model_name_pretrained.split('/')[-1].split('-')[0])
            train_epochs            = train_epochs - train_epochs_pretrained
            print("Pre-trained model in '%s' is loaded. Training is continued at epoch %s." % (model_name_pretrained, train_epochs_pretrained))
            for file_ in ['history.txt', 'weights_last.h5', 'weights_best.h5']:
                copy_file( source_file = file_,
                           source_dir  = model_name_pretrained,
                           target_dir  = '%s/%s' % (basedir, model_name),
                           overwrite_target = True )
            for file_ in ['evaluation.png', 'evaluation2.png']:
                copy_file( source_file = file_,
                           target_file = 'pre-'+file_,
                           source_dir  = model_name_pretrained,
                           target_dir  = '%s/%s' % (basedir, model_name),
                           overwrite_target = True )
        if continue_training:
            if os.path.exists('%s/models/%s/callbacks' % (basedir, model_name)):
                callback_file = [item for item in os.listdir('%s/%s/callbacks' % (basedir, model_name)) if 'hdf5' in item]
                if callback_file != []:
                    for file_ in callback_file:
                        os.rename( '%s/%s/callbacks/%s'     % (basedir, model_name, file_) ,
                                   '%s/%s/callbacks/pre-%s' % (basedir, model_name, file_) )
                    callback_file = sorted(callback_file)[-1]
                    train_epochs_pretrained = int(callback_file.split(".")[0].split('_')[-1])
                    train_epochs            = train_epochs - train_epochs_pretrained
                else:
                    continue_training = False
            else:
                continue_training = False

        if train_epochs<=0:
            print("'train_epochs<=0' - return without training.")
            return
    
        c = axes_dict(axes)['C']
        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

        config = Config(axes, n_channel_in, n_channel_out,
                        unet_n_depth          = 3,
                        train_batch_size      = train_batch_size,
                        train_steps_per_epoch = train_steps_per_epoch,
                        train_epochs          = train_epochs,
                        train_loss            = 'laplace'  if train_loss == 'probabilistic' else train_loss,
                        probabilistic         = True       if train_loss == 'probabilistic' else False,
                        ms_ssim_no_weights    = ms_ssim_no_weights  if train_loss == 'ms_ssim' else None,
                        ms_ssim_filter_size   = ms_ssim_filter_size if train_loss == 'ms_ssim' else None,
                        train_learning_rate   = train_learning_rate,
                        callback_period       = callback_period
                       )
        print(config)
        vars(config)
        
        ### change directory to avoid errors due to long directory names

        model = CARE(config, model_name, basedir = basedir)

        if not model_name_pretrained is None:
            model.load_weights('weights_best.h5')
            print("Weights are loaded from pre-trained model.")
        elif continue_training:
            model.load_weights('callbacks/%s' % callback_file)
            print("Weights are loaded from earlier callback: %s" % callback_file)

        history = model.train(X, Y, validation_data = (X_val,Y_val))

        cols = list(history.history.keys())    

        plt.figure(figsize=(16,5))
        plot_history(history,[cols[4],cols[0]],[cols[5],cols[1],cols[6],cols[2], cols[7],cols[3]]);
        plt.savefig('%s/loss.png' % folder_out)
        plt.show()

        plt.figure(figsize=(12,7))
        _P = model.keras_model.predict(X_val[:5])
        if config.probabilistic:
            _P = _P[...,:(_P.shape[-1]//2)]
        plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
        plt.suptitle('5 example validation patches\n'      
                     'top row: input (source),  '          
                     'middle row: target (ground truth),  '
                     'bottom row: predicted from source');
        plt.savefig('%s/pred.png' % folder_out)
        plt.show()

        if (continue_training is False) and (model_name_pretrained is None):
            history_dict = {'epoch no': list(range(1,train_epochs+1)),
                            **history.history}
        else:
            history_dict = {'epoch no': list(range(train_epochs_pretrained+1,train_epochs+train_epochs_pretrained+1)),
                            **history.history}

        cols.insert(0,'epoch no')

        if not os.path.exists('%s/history.txt' % folder_out):
            file = open("%s/history.txt" % folder_out,"a")
            file.write("\t".join(cols))
            file.write("\n")
            file.close()

        file = open("%s/history.txt" % folder_out,"a")    
        for i in range(len(history_dict[cols[0]])):
            for j in range(len(cols)):
                file.write(str(history_dict[cols[j]][i])+"\t")
            file.write("\n")
        file.close()
    
    return folder_out
    

def prediction(x, gt=None, model_name=None, thres=None, axes='YX', return_all=False, verbose=False, n_tiles=None):
    
    """Prediction in CSBDeep Framework.

    Parameters
    ----------
    x          : numpy array; input image
    gt         : 'None' or numpy array; ground truth image
    model_name : str; path to model
    thres      : 'None' or list with lower and upper percentiles
    axes       : str
    return_all : boolean; return all or only restored image?
    verbose    : boolean; print figures?
    n_tiles    : 'None' or tuple
    """
    
    if not thres is None:
        x = im(x).adjust(lower=thres[0], upper=thres[1])
        if not gt is None:
            gt = im(gt).adjust(lower=thres[0], upper=thres[1])
    
    model    = CARE(config=None, name=model_name, basedir='')
    restored = model.predict(x, axes, normalizer = None, n_tiles=n_tiles)
    
    if verbose:
        plt.figure(figsize=(16,10))
        plot_some([x, restored, gt] if not gt is None else [x, restored],
                  title_list=[['Input', 'Predicted', 'GT'] if not gt is None else ['Input', 'Predicted']],
                  pmin=0, pmax=100);
    
    if return_all:
        if not gt is None:
            return [x, restored, gt]
        else:
            return [x, restored]
    else:
        return restored