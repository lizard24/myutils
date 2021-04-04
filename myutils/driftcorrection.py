import numpy as np
import gc
import random

from csbdeep.utils import plot_some
from csbdeep.data  import RawData, create_patches, norm_percentiles
from csbdeep.io    import save_tiff_imagej_compatible

from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import center_of_mass

from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

from myutils.myimageprocess import im, norm_minmse


##################### Metrics #####################

def _mse(X, Y):
    return np.average(np.square( X - Y ))

def _ssim(X, Y): 
    return ssim(X, Y, data_range=1, full=False, gaussian_weights=True, sigma=1.5, win_size=11)

def _ms_ssim(X, Y, scale=1, filter_size=None, power_factors=None):
    ### defaults: filter_size=11, power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    ### last dimensions of X and Y are height, width, channel
    
    from tensorflow.image     import ssim_multiscale as ms_ssim
    from tensorflow           import convert_to_tensor
    from tensorflow.compat.v1 import Session
    
    if scale==1:
        if power_factors is None: power_factors = [1]
        if filter_size   is None: filter_size   = 11
    elif scale==3:
        if power_factors is None: power_factors = (0.2096, 0.4659, 0.3245)
        if filter_size   is None: filter_size   = 11
    elif scale==5:
        if power_factors is None: power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
        if filter_size   is None: filter_size   = 7
    
    if len(X.shape)==2: X, Y = np.reshape(X, X.shape+(1,)), np.reshape(Y, Y.shape+(1,))
        
    X, Y = convert_to_tensor(X), convert_to_tensor(Y)
    
    sess = Session()
    ms_ssim_ = ms_ssim(X, Y, 1, power_factors=power_factors, filter_size=filter_size).eval(session=sess)       
    sess.close()
    gc.collect()
    
    return ms_ssim_

##################### Correlation Map/Offset #####################

def find_coordinates(array, mode):
    if mode == 'cm':
        (x,y) = center_of_mass(array)
        return (int(x),int(y))
    elif mode == 'min':
        return np.unravel_index(np.argmin(array), array.shape)
    elif mode == 'max':
        return np.unravel_index(np.argmax(array), array.shape)


def get_correlation_map(benchmark, img, metric, shuffle):
    ### X and Y come as CTYX with C=patch_number, T=1
    
    def get_metric(X, Y, metric):
        ### X and Y come as CYX
        ### returns C-dim vector
        if metric in ['mse', 'ssim']:
            metric_ = np.empty(X.shape[0])
            for n in range(X.shape[0]):
                metric_[n] = eval('_%s' % metric)(X[n], Y[n])
            return metric_
        else:
            X = np.reshape(X, X.shape+(1,))
            Y = np.reshape(Y, Y.shape+(1,))
            return _ms_ssim(X, Y, scale=int(metric[-1]))
    
    
    sliding_window = 2*shuffle+1
    width = img.shape[-1] - 2*shuffle
    
    benchmark = benchmark[:,0,shuffle:-shuffle,shuffle:-shuffle]
    
    data_x = np.empty((sliding_window, sliding_window)+benchmark.shape)
    data_y = np.empty((sliding_window, sliding_window)+benchmark.shape)
    
    
    for i in range(sliding_window):
        for j in range(sliding_window):
            data_x[i,j] = benchmark
            data_y[i,j] = img[:, 0, i:i+width, j:j+width]
        
    data_x = np.reshape(data_x, (np.prod(data_x.shape[:3]),)+data_x.shape[3:])
    data_y = np.reshape(data_y, (np.prod(data_y.shape[:3]),)+data_y.shape[3:])
     
    map_ = get_metric(data_x, data_y, metric)
    map_ = np.reshape(map_, (sliding_window, sliding_window, benchmark.shape[0]))
        
    return np.rot90(np.mean(map_, axis=2, keepdims=True)[:,:,0], k=2) ### no idea why rotation is necessary, prob artefact from reshaping


def realign(benchmark, img, shuffle, N, patch_size, metric='mse', display=True, norm=True):
     
    def no_norm(patches_x, patches_y, x, y, mask, channel):
        return patches_x, patches_y
    
    raw_data = RawData.from_arrays( [im(gaussian(benchmark, 2)).adjust(lower=2, upper=99.8) if norm else benchmark],
                                    [im(gaussian(img, 2)).adjust(lower=2, upper=99.8) if norm else img] )
    
    patch_size=patch_size+2*shuffle
    
    if (metric=='ms-ssim3') and (patch_size < 48): patch_size=48
    
    ### X and Y come as CTYX
    X, Y, XY_axes = create_patches (raw_data   = raw_data,
                                    patch_size = (patch_size, patch_size),
                                    n_patches_per_image = N,
                                    verbose    = False,
                                    normalization = norm_percentiles() if norm else no_norm,)
    
    if norm:
        for n in range(X.shape[0]):
            X[n,0], Y[n,0] = norm_minmse(X[n,0], Y[n,0], normalize_gt=True)
    
    correlation_map=np.asarray([[None]*(2*shuffle+1)]*(2*shuffle+1), dtype=float)

    cx, cy, cx_, cy_, n = shuffle, shuffle, 0, 0, 0

    while (n<shuffle) and ((cx_, cy_ != (1,1))):
        correlation_map[cx-1:cx+2,cy-1:cy+2] = get_correlation_map(X[...,shuffle:-shuffle,shuffle:-shuffle],
                                                                   Y[...,2*shuffle-cx:-cx,2*shuffle-cy:-cy],
                                                                   metric,
                                                                   shuffle=1)
        (cx_, cy_) = find_coordinates(correlation_map[cx-1:cx+2,cy-1:cy+2], 'min' if metric in ['mse'] else 'max')        
        cx, cy, n = cx+cx_-1, cy+cy_-1, n+1
    
    if display:
        correlation_map[cx, cy]                    = eval('np.nan%s' % ('max' if metric in ['mse'] else 'min'))(correlation_map)
        correlation_map[np.isnan(correlation_map)] = eval('np.nan%s' % ('max' if metric in ['mse'] else 'min'))(correlation_map)
        plt.figure(figsize=(16,10))
        plot_some([correlation_map], title_list=[[metric]], pmin=0,pmax=100);
        
    return [shuffle-cx, shuffle-cy]


##################### Main Function #####################

def correctdrift(data,
                 benchmark    = None,
                 shuffle      = 10,
                 patch_number = 50,
                 patch_size   = 50,
                 crop         = None,
                 metric       = 'mse',
                 display      = False,
                 execute      = True,
                 norm         = True,
                 axes         = 'TYX'):
    
    """lateral drift correction of a numpy array

    Parameters
    ----------
    data          : 3- or 4-dim numpy array
    benchmark     : 'None' or numpy array; use to calculate offsets based on different channel and/or cropped ROI
    shuffle       : int; shuffle size in px; it's advisable to use twice the expected image displacement
    patch_number  : int; number of image patches
    patch_size    : int; patch_size in px
    crop          : crop size; either 'int' or None
    metric        : 'mse', 'ssim', 'ms_ssim3'
    display       : boolean; displays correlation maps if True
    execute       : boolean; returns corrected array if True, returns offset vector if False
    norm          : boolean; contrast adjusts the benchmark to (2, 99.8) before calculating offsets
    axes          : supports 'TYX', 'CTYX' and 'TCYX'
    
    Return
    ------
    corrected data if execute=True, otherwise list of offset vectors
    
    """
        
    
    def finddrift3d(data3d, dx_initial=None, dy_initial=None):
        
        dx = np.zeros((tt,), dtype=np.int)
        dy = np.zeros((tt,), dtype=np.int)
        
        xx, yy = data3d.shape[1:]
        
        benchmark_ = data3d[0, shuffle+dx[0]:xx-shuffle+dx[0],shuffle+dy[0]:yy-shuffle+dy[0]]
        
        for t in range(1, tt):
            
            dx_ini = dx[t-1] if dx_initial is None else dx_initial[t]
            dy_ini = dy[t-1] if dy_initial is None else dy_initial[t]
            
            img = data3d[t, shuffle+dx_ini:xx-shuffle+dx_ini, shuffle+dy_ini:yy-shuffle+dy_ini]
            
            dd = realign(benchmark_, img, shuffle, patch_number, patch_size, metric=metric, display=display, norm=norm)
            
            dx[t], dy[t] = dd[0]+dx_ini, dd[1]+dy_ini
            
        return [dx, dy]
       
    
    if not (axes in ['TYX', 'CTYX', 'TCYX']) or (len(axes)!=len(data.shape)):
        print("Wrong axes. Data is returned without correction.")
        return data
    
    if benchmark is None: benchmark = data
    
    if axes=='TYX':
        data      = np.reshape(data, (1,) + data.shape)
        benchmark = np.reshape(benchmark, (1,) + benchmark.shape)
    elif axes=='TCYX':
        data      = np.swapaxes(data, 1, 0)
        benchmark = np.swapaxes(benchmark, 1, 0)
        
    
    cc, tt = data.shape[:2]
        
    offset_dx, offset_dy = [None]*data.shape[0], [None]*data.shape[0]

    for c in range(cc):
        [offset_dx[c], offset_dy[c]] = finddrift3d(benchmark[c],
                                                   dx_initial = offset_dx[c-1] if c!=0 else None,
                                                   dy_initial = offset_dy[c-1] if c!=0 else None)

    if cc>1:
        for c in range(1, cc):
            dd = realign( np.mean(benchmark[c-1], axis=0), np.mean(benchmark[c], axis=0),
                          shuffle, patch_number, patch_size, metric=metric, display=display, norm=norm )
            offset_dx[c] += dd[0]
            offset_dy[c] += dd[1]

    if execute:
        
        if crop is None: crop=shuffle+1
        
        new_data = np.copy(data[...,crop:-crop, crop:-crop])
        
        xx, yy = data.shape[2:]

        for c in range(cc):
            for t in range(tt):
                if display: print((offset_dx[c][t], offset_dy[c][t]))
                new_data[c,t] = data[c,t,crop+offset_dx[c][t]:xx-crop+offset_dx[c][t],crop+offset_dy[c][t]:yy-crop+offset_dy[c][t]]
        
        
        if axes=='TYX':
            new_data = np.reshape(new_data, new_data.shape[1:])
        elif axes=='TCYX':
            new_data = np.swapaxes(new_data, 0, 1)

        return new_data

    else:
        return offset_dx, offset_dy

    
##################### Test Main Function #####################

def test_correctdrift(data=None, dd=None, shuffle=5, metric='mse', display=False, save=False):
    
    """test of function 'correctdrift'
    
    Parameters
    ----------
    data         : 2-dim numpy array; if 'None' binary testdata is created
    dd           : 2-dim tuple; displacement vector; is created randomly if 'None'
    shuffle      : int; shuffle size in px (see 'correctdrift')
    metric       : 'ssim', 'ms_ssim3', 'ms_ssim5' or 'mse'
    display      : boolean; displays correlation maps if True
    save         : saves testdata and corrected data as tiff files
    
    Return
    ------
    tiff files if save=True; prints status of test result (success/fail)
    
    """
    
    def create_testdata(data=None, data_shape=(2000,2000), thres=0.8, dd=None):

        if data is None:
            testdata = np.random.rand(data_shape[0], data_shape[1])
            testdata[testdata>thres] = 1
            testdata[testdata<=thres] = 0
        else:
            testdata=data
            
        crop = np.max(np.abs(dd))*2 if np.max(np.abs(dd))*2 !=0 else 1

        return np.stack( ( np.stack( (testdata[2*crop        :-2*crop+1         , 2*crop        :-2*crop+1       ],
                                      testdata[2*crop+dd[0]  :-2*crop+dd[0]+1   , 2*crop+dd[1]  :-2*crop+dd[1]+1 ]) ),
                           np.stack( (testdata[2*crop+dd[0]  :-2*crop+dd[0]+1   , 2*crop+dd[1]  :-2*crop+dd[1]+1 ],
                                      testdata[2*crop+2*dd[0]:-2*crop+2*dd[0]+1 , 2*crop+2*dd[1]:-2*crop+2*dd[1]+1]) ) ) )
    
    dd = (random.randint(1,shuffle), random.randint(1,int(shuffle/2))) if dd is None or (dd[0]>=shuffle) or (dd[1]>=shuffle) else dd
    
    testdata = create_testdata(data=data, dd=dd)
    data     = correctdrift(testdata, shuffle=shuffle, axes='CTYX', metric=metric, display=display)
    
    if np.mean(np.sum(data, axis=0)[0]-np.sum(data, axis=0)[1]) != 0:
        print("Test failed.")
    else:
        print("Test succeeded.")
        
    if save:
        save_tiff_imagej_compatible('testdata_corrected.tif', data, axes='CTYX')
        save_tiff_imagej_compatible('testdata.tif', testdata, axes='CTYX')