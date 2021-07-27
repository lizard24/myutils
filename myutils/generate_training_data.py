import numpy as np
import os

import matplotlib.pyplot as plt

from csbdeep.data  import RawData, norm_percentiles, no_background_patches, create_patches, shuffle_inplace
from csbdeep.io    import load_training_data, save_training_data
from csbdeep.utils import plot_some

from myutils.general         import generate_folder
from myutils.myimageprocess  import im


def create_npz( basepath     = '',                  ### strings
                source_dir   = None,                ### string
                target_dir   = None,                ### string
                patch_size   = None,                ### tuple
                n_patches_per_image = None,         ### int
                axes         = 'YX',                ### string
                save_file    = 'traig-data.npz',    ### string
                norm         = False,               ### False or tuple with (pmin, pmax)
                patch_filter = True ):              ### boolean

    def no_norm(patches_x, patches_y, x, y, mask, channel):
        return patches_x, patches_y

    if norm is False:
        norm=no_norm
    elif norm is True:
        norm = norm_percentiles()
    else:
        norm = norm_percentiles(percentiles=norm)
    
    raw_data = RawData.from_folder (basepath    = basepath,
                                    source_dirs = [source_dir],
                                    target_dir  = target_dir,
                                    axes        = axes)
    
    generate_folder(save_file, level=-1)
    
    save_file_=save_file
    
    create_patches( raw_data            = raw_data,
                    save_file           = save_file_,
                    patch_size          = patch_size,
                    n_patches_per_image = n_patches_per_image,
                    patch_filter        = no_background_patches() if patch_filter else None,
                    normalization       = norm,
                    verbose             = True )
    



def merge_npz( files_npz  = None  ,     ### list of npz files
               save_file  = None  ,
               verbose    = False ,
               delete     = False   ):   ### string
    
    if type(files_npz) is str:
        files_npz = [files_npz]
    
    for n, file in enumerate(files_npz):
        (X_, Y_), (X_val, Y_val), axes = load_training_data( file, validation_split=0, verbose=False )
        if n==0:
            X, Y = X_, Y_
        else:
            X = np.concatenate((X, X_), axis=0)
            Y = np.concatenate((Y, Y_), axis=0)
    
    X, Y = list(X), list(Y)

    shuffle_inplace(X,Y)
    
    X, Y = map(lambda x: np.asarray(x), (X, Y))
    
    generate_folder(save_file, level=-1)
    
    save_training_data( save_file+'.npz' if not '.npz' in save_file else save_file,
                        X, Y, axes )
    
    if delete:
        for file in files_npz:
            os.remove(file)
    
    if verbose:
        for i in range(2):
            plt.figure(figsize=(16,4))
            sl = slice(8*i, 8*(i+1)), 0
            plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
            plt.savefig(save_file.replace('npz', 'png'))
            plt.show()