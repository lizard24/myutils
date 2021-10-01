import numpy as np
import math
import gc


from skimage.metrics import structural_similarity as ssim  
from myutils.myimageprocess import im, norm_minmse


## Mean Absolute Error
def mae(img_i, img_f, norm=False):
    
    if norm==True:
        img_i, img_f = norm_minmse(img_i, img_f, normalize_gt=True)
        
    A    = img_f-img_i
    mae_ = np.average( A )
    
    return mae_

## Mean Square Error
def mse(img_i, img_f, norm=False):
    
    if norm==True:
        img_i, img_f = norm_minmse(img_i, img_f, normalize_gt=True)
    
    A    = img_f-img_i
    mse_ = np.average( np.square(A) )
    
    return mse_


############ CONTRAST

## Michelson Contrast
def mc(img, norm=False, mask=None):
    
    if norm==True:
        img = im(img).adjust(lower=2, upper=99.8)
    
    Imax = 1
    
    if mask is None:
        Imin = np.median(img)
    else:
        intensities = img[img*mask>0]
        Imin = np.min(intensities)
        
    return ( Imax - Imin ) / ( Imax + Imin )

## Weber Contrast
def wc(img, norm=False, mask=None):
    
    if norm==True:
        img = im(img).adjust(lower=2, upper=99.8)
        
    if mask is None:
        intensities = np.sort(np.reshape(img, np.prod(img.shape)))
        intensities = intensities[intensities>=np.median(test)]
    else:
        intensities = img[img*mask>0]
        
    Ilum = intensities.mean()
    Ibg  = np.percentile(img, 5)
    
    return (Ilum - Ibg) / Ibg

### Root Mean Squared Contrast
def rmsc(img, norm=False, mask=None):
    
    if norm==True:
        img = imadjust(img, 2, 99.8, 1)
       
    if mask is None:
        intensities = np.sort(np.reshape(img, np.prod(img.shape)))
        intensities = intensities[intensities>=np.median(test)]
    else:
        intensities = img[img*mask>0]
    
    return intensities.std()


############ SIMILARITY METRICS

## Normalised Root Mean Squared Error
def nrmse(img_i, img_f, data_range=1, norm=False, **kwargs):

    if norm==True:
        img_i, img_f = norm_minmse(img_i, img_f, normalize_gt=True)
        
    return np.sqrt( mse(img_i, img_f, norm=norm) ) / data_range

## Peak Signal-to-Noise Ratio between the Images
def psnr(img_i, img_f, data_range=1, norm=False, **kwargs):
    
    if norm==True:
        img_i, img_f = norm_minmse(img_i, img_f, normalize_gt=True)
    
    psnr_ = 10 * math.log10( np.square(data_range) / mse(img_i, img_f, norm=norm) )
    
    ### alternative:
#     from skimage.metrics import peak_signal_noise_ratio as psnr
#     psnr(img_i, img_f, data_range=data_range)
    
    return psnr_


def ms_ssim(X, Y, data_range=1, scales=1, filter_size=11, power_factors=None, filter_sigma=1.5, mean=False):
    
    from tensorflow.image import ssim_multiscale as _ms_ssim
    import tensorflow as tf

    if (scales==1) and (power_factors is None):
        power_factors = [1]
    elif (scales==3) and (power_factors is None):
        power_factors = [0.2096, 0.4659, 0.3245]
    elif (scales==5) and (power_factors is None):
        power_factors = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        
        
    if len(X.shape)==2:
        X, Y = map(lambda x: np.reshape(x, (1,)+x.shape), (X, Y))
    if len(X.shape)==3:
        X, Y = map(lambda x: np.reshape(x, x.shape(1,)), (X, Y))

        
    ms_ssim_ = _ms_ssim(X, Y, data_range, power_factors=power_factors, filter_size=filter_size, filter_sigma=filter_sigma, k1=0.01, k2=0.03)

    if mean:
        ms_ssim_ = tf.reduce_mean(ms_ssim_)
        
    with tf.Session() as sess:
        ms_ssim_ = ms_ssim_.eval()
    sess.close() 
    gc.collect()

    return ms_ssim_


############ SIMILARITY METRICS + MAPS

## SSIM Metric + Map
def ssim(returns, img_i, img_f, data_range=1, norm=False, gaussian_weights=True, sigma=1.5, win_size=None, **kwargs):
    ### idx is the mean structural similarity over the img
    ### map is the full SSIM
    ### as far as I can see, there is a mistake in ZeroCostDL4:
    ##### "The SSIM maps are constructed by calculating the SSIM metric in each pixel by considering the
    ##### surrounding structural similarity in the neighbourhood of that pixel (currently defined as window
    ##### of 11 pixels and with Gaussian weighting of 1.5 pixel standard deviation"
    ### but then don't specify in function accordingly (gaussian_weights=True, sigma=1.5)
    ### gaussian_weights=False as default
    ### I don't see how they can specify the win_size since:
    ### win_size is the side-length of the sliding window used in comparison. By default, it is None (def not 11)
    ### if gaussian_weights=True, win_size is ignored and the window size will depend on sigma.
    
    ### old:
    ### from skimage.measure import compare_ssim as ssim
    ### idx, map = ssim(img_i, img_f, full=True)
    
    if norm==True:
        img_i, img_f = norm_minmse(img_i, img_f, normalize_gt=True)  
    
    if (returns == 'both') or (returns == 'map'):
        idx_, map_ = ssim(img_i, img_f, data_range=data_range, full=True, gaussian_weights=gaussian_weights, sigma=sigma, win_size=win_size)
        
        if returns == 'both':
            return idx_, map_
        elif returns == 'map':
            return map_
    else:
        idx_ = ssim(img_i, img_f, data_range=data_range, full=False, gaussian_weights=gaussian_weights, sigma=sigma, win_size=win_size)
        return idx_


## NCC Metric + Map
def ncc(returns, img_i, img_f, mask=np.ones((3,3)), norm=False, **kwargs):
    
    ## img is a list - if two entries: img[0]=img_f, img[1]=img_i
    ## mask is a binary nxn array denoting the subarea
    ## does not calculate border area of image
        
    ## Normalized Cross-Correlation  ==  Pearson Correlation Coefficient
    def ncc_idx(x, y):
        ### x, y are either 1D or 2D arrays

        x = x - np.average(x)
        y = y - np.average(y)

        A = np.sum( x * y )

        x = np.square(x)
        y = np.square(y)

        B = np.sqrt( np.sum(x) ) * np.sqrt( np.sum(y) )

        ncc_ = 1 if A==B else A/B

        return ncc_
    
    if norm==True:
        img_i, img_f = norm_minmse(img_i, img_f, normalize_gt=True)
    
    if returns != 'idx':

        map_ = np.zeros(img_i.shape)

        a0 , a1 = int((mask.shape[0]-1)/2) , int((mask.shape[1]-1)/2)
        (rows, cols) = img_i.shape

        idx = np.where(mask==1)

        for i in range(a0,rows-a0-1):
            for j in range(a1,cols-a1-1):

                img_ii = img_i[ i-a0:i+a0+1, j-a1:j+a1+1 ]
                img_ff = img_f[ i-a0:i+a0+1, j-a1:j+a1+1 ]

                map_[i,j] = ncc_idx(img_ii[idx], img_ff[idx])

    if returns != 'map':
        ncc_ = ncc_idx(img_i, img_f)
    
    if returns == 'idx':
        return ncc_
    elif returns == 'map':
        return map_
    else:
        return ncc_ , map_

    
### Root Mean Squared Error Metric + Map
def rms(returns, img_i, img_f, norm=False):
    
    if norm==True:
        img_i, img_f = norm_minmse(img_i, img_f, normalize_gt=True)

    rms_ = np.sqrt(np.square(img_i - img_f))
    
    if returns == 'idx':
        return np.average(rms_)
    elif returns == 'map':
        return rms_
    elif returns == 'both':
        return np.average(rms_), rms_
