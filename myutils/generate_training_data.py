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
    
#     n, save_file_ = 0, ''
#     while (os.path.exists(save_file_)) or (n<1):
#         save_file_ = "%s_%04d.npz" % (save_file.split('.')[0], n)
#         n+=1
    save_file_=save_file
    
    create_patches( raw_data            = raw_data,
                    save_file           = save_file_,
                    patch_size          = patch_size,
                    n_patches_per_image = n_patches_per_image,
                    patch_filter        = no_background_patches() if patch_filter else None,
                    normalization       = norm,
                    verbose             = True )
    


def merge_npz( folder_npz = None,     ### string
               save_file  = None ):   ### string
     
    filenames = [item for item in os.listdir(folder_npz) if '.npz' in item]
    
    X, Y = [None]*len(filenames), [None]*len(filenames)
    
    for n, file in enumerate(filenames):
        (X[n], Y[n]), (X_val, Y_val), axes = load_training_data(folder_npz+'/'+file, validation_split=0, verbose=False)
    
    def list_to_array(liste):
        data = liste[0]
        for item in liste[1:]:
             data = np.concatenate((data,item), axis=0)
        return data
        
    X, Y = list_to_array(X), list_to_array(Y)
    shuffle_inplace(X,Y)
    X, Y = np.swapaxes(X, 1, 3), np.swapaxes(Y, 1, 3)
    
    generate_folder(save_file, level=-1)
              
    save_file = save_file+'.npz' if not '.npz' in save_file else save_file

    save_training_data(save_file, X, Y, axes)
    
    for i in range(2):
        plt.figure(figsize=(16,4))
        sl = slice(8*i, 8*(i+1)), 0
        plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
        plt.savefig(save_file.replace('npz', 'png'))
        plt.show()
