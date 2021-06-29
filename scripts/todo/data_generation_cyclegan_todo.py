import numpy as np
from csbdeep.io import load_training_data
from myutils.general import generate_folder
from tifffile import imsave


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="write report to FILE", metavar="FILE")
parser.add_argument("-q", "--quiet",
                    action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")

args = parser.parse_args()


aligned = False

dataset_name = 'pax_L2L'

dir_data = "I:/Science/SIPBS/McConnellG/LisaKoelln/Paper/2020_Label2label/data/200820_Paxillin/200925_paxillin/model/l2l_128px_256-256-1024-256-256_norm-1-99.95_aug-0.5-0.75-1-1.25-1.5_rot-1"


####  npz files come in STYXC or SYXC, data range between 0 and 1
####  for unaligned: AtoB training: input left (A), GT right (B)
####  for aligned: AtoB training as in "horse2zebra" - here "pfa2glyx" - A=pfa, B=glyx

if run:

    def solve(data, folder):
        generate_folder(folder)
        for n in range(data.shape[0]):
            export = np.uint8(data[n,...,0]*255)
            imsave('%s/%s.jpg' % (folder, n), export)

    (X, Y), (X_val, Y_val), axes = load_training_data(dir_data+'/traig-data.npz', validation_split=0.05, verbose=True)
   
    print(X.shape)
    print(axes)
    print(np.max(X[0]))
   
    if aligned:

        solve( X     , 'datasets/%s/trainA' % dataset_name )
        del X
        solve( X_val , 'datasets/%s/testA'  % dataset_name )
        del X_val
        solve( Y     , 'datasets/%s/trainB' % dataset_name )
        del Y
        solve( Y_val , 'datasets/%s/testB'  % dataset_name )
        del Y_val
       
    else:
       
        X = np.concatenate([X, Y], axis=axes.find('X'))
        del Y
        solve( X     , 'datasets/%s/train'  % dataset_name )
        del X
        X_val = np.concatenate([X_val, Y_val], axis=axes.find('X'))
        del Y_val
        solve( X_val , 'datasets/%s/test'   % dataset_name )
        del X_val


    gc.collect()