import numpy as np
import matplotlib.pyplot as plt

import cv2
from cv2_rolling_ball import subtract_background_rolling_ball

from scipy.ndimage    import gaussian_filter
from scipy.ndimage    import gaussian_filter1d
from scipy.signal     import argrelextrema
from scipy            import interpolate

from csbdeep.utils import plot_some

def imshape2d(file):
    from PIL import Image
    return Image.open(file).size


class im:
    
    ################## INITIALISE ##################
    
    def __init__(self, data):
        
        self.dims_first = data.shape[:-2]
        self.dims_last  = data.shape[-2:]
        
        self.N = 1 if self.dims_first == () else np.prod(self.dims_first)
        
        self.data = np.reshape(data, (self.N,) + self.dims_last)

    ################## EXIT ##################
    
    def exit(self):
        
        return np.reshape(self.data, self.dims_first + self.dims_last)
        
     
    ################## FUNCTIONS OUTSIDE OF INIT AND EXIT ##################


    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
    def im2rgb(blue    = None,
               red     = None,
               green   = None,
               dtype   = None,
               display = False):
        
        ## blue, red, green have to be 2dim arrays
        
        for item in [blue, red, green]:
            if not item is None: dims = item.shape
        
        rgb = np.zeros(dims+(3,))
        if not blue  is None: rgb[...,0] = blue
        if not red   is None: rgb[...,1] = red
        if not green is None: rgb[...,2] = green
          
        
        if (dtype is None) and not ('uint' in str(rgb.dtype)):
            rgb = im.uint8(rgb)

        if display:
            plt.figure(figsize=(16,10))
            plot_some([cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)], pmin=2,pmax=99.8);
            
        return rgb
    
    
    def uint8(data, norm=True):
        if norm: data = im(data).norm(glob=True)
        data = np.uint8(data*255)
        return data

    def uint16(data, norm=True):
        if norm: data = im(data).norm(glob=True)
        data = np.uint16(data*65535)
        return data

    def uint32(data, norm=True):
        if norm: data = im(data).norm(glob=True)
        data = np.uint8(data*4294967295)
        return data

    def uint64(data, norm=True):
        if norm: data = im(data).norm(glob=True)
        data = np.uint16(data*18446744073709551615)
        return data
        
    def gethist(data,
                histSize   = 256,
                histRange  = (0, 256),
                accumulate = False,
                dtype      = 'uint8'):
        
        """Note: returns histogram and bins
        """
        
        dims_first = data.shape[:-2]
        dims_last  = data.shape[-2:]
        
        N = 1 if dims_first == () else np.prod(dims_first)
        
        data = np.reshape(data, (N,1,) + dims_last)
        
        if 'float' in dtype: dtype = 'uint8'
            
        if dtype != str(data.dtype):
            data = eval('im.%s' % dtype)(data)
              
        histogram = []
        for n in range(N):
            histogram.append( cv2.calcHist(data[n], [], None, [histSize], histRange, accumulate=accumulate) )
            histogram[n] = np.asarray([np.int(item) for item in histogram[n]])
        
        hist = np.reshape( np.asarray(histogram), dims_first+(histSize,) )
        bins = np.asarray(range(histRange[0], histRange[1], int((histRange[1]-histRange[0])/histSize)))
        
        return hist, bins
    

    ################## FUNCTIONS THAT RETURN IMAGES ##################

    def delete_nan(self):
        
        def fct(img):
            if np.nanmin(img) >= 0:
                img = np.nan_to_num(img, copy=True, nan=np.nanmin(img), posinf=None, neginf=None)
            else:
                img = np.zeros(img.shape)
    
        for n in range(self.N):
            self.data[n] = fct(self.data[n])
            
        return im.exit(self)
    
    def norm(self, minVal=0, maxVal=1, glob=False):
        self.data = np.float64( (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data) ) )
        if (glob==False) and (self.N>1):
            for n in range(self.N):
                self.data[n] = (self.data[n] - np.min(self.data[n])) / (np.max(self.data[n]) - np.min(self.data[n]))
        self.data = self.data * (maxVal - minVal) + minVal
        return im.exit(self)
    
    def adjust(self, lower=0, upper=100, absolute=False, minVal=0, maxVal=1, clip=True, glob=False):

        """Contrast adjust a 1/n-dim numpy array along the last 1/2 dimensions.
        In default, function normalises array to (0, 1)

        Parameters
        ----------
        data     : numpy array
        lower    : either value or percentile; percentile between 0-100 if absolute=False
        upper    : either value or percentile; percentile between 0-100 if absolute=False
        absolute : boolean; to adjust to absolute lower/upper values
        minVal   : final minimum value; set to 'None' to disable normalisation
        maxVal   : final maximum value; set to 'None' to disable normalisation
        clip     : boolean; sets values outside of (minVal, maxVal) to (0, 1)

        Return
        ------
        intensity adjusted array
        
        Note
        ------
        from skimage import exposure
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

        """

        def imadjust(img):
            if (lower>0) or (upper<100):
                lower_int = lower if absolute else np.percentile(img, lower)
                upper_int = upper if absolute else np.percentile(img, upper)
                img = (img - lower_int) / (upper_int - lower_int)
                if clip: img = np.clip(img, 0, 1)
            return im(img).norm(minVal=minVal, maxVal=maxVal, glob=True)
        
        self.data = im(self.data).norm(glob=False)
        
        if glob:
            self.data = imadjust(self.data)
        else:
            for n in range(self.N):
                self.data[n] = imadjust(self.data[n])
           
        return im.exit(self)
    
    
    def standardisation(self, mean=None, sigma=None, clip=True, glob=True):
        
        self.data = np.float64(self.data/np.max(self.data))
        
        if glob:
	    sigma = sigma if not sigma is None else np.std(self.data)
            mean  = mean  if not sigma is None else np.mean(self.data)
            self.data = (self.data - np.mean(self.data)) / np.std(self.data)
	    self.data = self.data * sigma + mean
        else:
            for n in range(self.N):
	        sigma_n = sigma if not sigma is None else np.std(self.data[n])
                mean_n  = mean  if not sigma is None else np.mean(self.data[n])
                self.data[n] = (self.data[n] - np.mean(self.data[n])) / np.std(self.data[n])
		self.data = self.data * sigma_n + mean_n


        if clip: self.data = np.clip(self.data, 0, 1)

        return im.exit(self)
    
    def gaussian(self, sigma=1):

        """Applies Gaussian filter along the last 1/2 dimensions.

        """
        
        for n in range(self.N):
            self.data[n] = gaussian_filter(self.data[n], sigma)
        
        return im.exit(self)

    def interpolate(self, new_dims=()):

        """interpolates the last 2 dimensions in a n-dim numpy array..

        Parameters
        ----------
        data     : numpy array
        new_dims : 2-dim tuple

        Return
        ------
        interpolated array

        """

        def interpolate2d(array):
            (xini, yini) = array.shape
            x = np.linspace(1, xini, xini, dtype=int)
            y = np.linspace(1, yini, yini, dtype=int)
            xnew = np.linspace(1, xini, new_dims[0], dtype=int)
            ynew = np.linspace(1, yini, new_dims[1], dtype=int)
            f = interpolate.RectBivariateSpline(x, y, array, bbox=[None, None, None, None], kx=3, ky=3, s=0)
            return f(xnew, ynew)
            
        if len(new_dims) != 2:
            print("Wrong new_dims - return data without interpolation")
            
        if self.dims_last != new_dims:

            data_new = np.empty((self.N,) + new_dims)
            
            for n in range(self.N):
                data_new[n] = interpolate2d(self.data[n])

            self.data      = data_new
            self.dims_last = data_new.shape[-2:]
            
        return im.exit(self)
            
    def flatfield(self, sigma=None, norm=True):

        """flatfield correction for the last 2 dimensions in a n-dim numpy array

        Parameters
        ----------
        data     : numpy array
        sigma    : sigma for the gaussian filter
        norm     : boolean; normalisation after flatfield correction?
        maxVal   : maximum value of the image - only applies if norm=True
        method   : 'subtract' or 'divide'


        Return
        ------
        flatfield corrected array

        """

        self.data = np.float64(self.data/np.max(self.data))
        
        for n in range(self.N):
            bg = np.sqrt(gaussian_filter(self.data[n], sigma))
            self.data[n] = self.data[n] / bg * np.mean(bg)
        
        if norm: self.data = im(self.data).norm(glob=True)
                
        return im.exit(self)
    
    def compdic(self, rot90=0, norm=True):
        
        self.data = np.float64(self.data/np.max(self.data))
        
        self.data = np.rot90(self.data, axes=(1,2), k=rot90)
        
        new_data = self.data[...,0:-2] - self.data[...,1:-1]
        self.data[:] = np.mean(new_data)
        self.data[...,0:-2] = new_data
        
        self.data = np.rot90(self.data, axes=(1,2), k=4-rot90)
        
        if norm: self.data = im(self.data).norm(glob=True)
            
        return im.exit(self)

    def clahe(self, clipLimit=2, tileGridSize=(8,8), dtype='uint8'):

        """Note: img has to be uint

        """
        
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        
        if not dtype in str(self.data.dtype):
            self.data = eval('im.%s' % dtype)(self.data)
        
        for n in range(self.N):
            self.data[n] = clahe.apply( self.data[n] )
        
        return im.exit(self)

    def tophat(self, filterSize=(11,11), dtype='uint8'):

        """Note: check if requirement for img!

        """
        
        filterSize = filterSize
        kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
        
        if not dtype in str(self.data.dtype):
            self.data = eval('im.%s' % dtype)(self.data)
           
        for n in range(self.N):
            self.data[n] = cv2.morphologyEx(self.data[n], cv2.MORPH_TOPHAT, kernel)
            
        return im.exit(self)

    def subtractbg(self, radius=25, light_background=False, use_paraboloid=True, do_presmooth=True, norm=True):

        """Note: img has to be uint8. Install via "pip install opencv-rolling-ball"

        """

        def subtractbg2d(img):
            img, background = subtract_background_rolling_ball( img, radius,
                                                                light_background = light_background,
                                                                use_paraboloid   = use_paraboloid,
                                                                do_presmooth     = do_presmooth )
            return img
        
        maximum = np.max(self.data)
        
        new_data = np.empty(self.data.shape)
        
        if not "uint8" in str(self.data.dtype):
            self.data = im.uint8(self.data)
        
        for n in range(self.N):
            new_data[n] = subtractbg2d( self.data[n] )
            
        self.data = np.array(new_data, dtype='float64') / 255 * maximum
        
        return im.exit(self)
    
   
    def my_hist_matching(self, histSize_ini=256, histSize_fin=256, display=False, mode='linear', tidy_hist=True ):

        self.data = im.uint8(self.data)
        histRange = (0, 256)
        interval  = int(256/histSize_fin)
            
        for k in range(self.N):
            
            img = np.copy(self.data[k])
                
            interval_ini = int(256/histSize_ini)
            
            hist, bins = im.gethist(img if interval_ini==1 else np.uint8(img/interval_ini)*interval_ini,
                                    histSize=histSize_ini,
                                    histRange=histRange)

            bins = np.delete(bins, np.where(hist==0))
            hist = np.delete(hist, np.where(hist==0))    
            if tidy_hist:
                for i in range(3):
                    bins = np.delete(bins, np.add(argrelextrema(hist[1:-1], np.less),1))
                    hist = np.delete(hist, np.add(argrelextrema(hist[1:-1], np.less),1))
            if not 255 in bins:
                hist = np.concatenate([hist, [len(np.where(img==255)[0])]])
                bins = np.concatenate([bins, [255]])

            interval_fin = int(256/histSize_fin)
            bins_new = np.asarray(range(0, 256, interval_fin))
            
            if mode=='spline':
                tck = interpolate.splrep(bins, hist, s=0)
                hist_new = interpolate.splev(bins_new, tck, der=0)
            else:
                f   = interpolate.interp1d(bins, hist, kind=mode)
                hist_new = f(bins_new)

            hist_new = gaussian_filter1d(hist_new, 1.5)
            hist_new[hist_new<0] = 0
            hist_new = hist_new / np.sum(hist_new) * np.prod(img.shape)
            hist_new = np.round(hist_new)
               
            if interval_fin!=1:
                img = np.uint8(img/interval_fin)*interval_fin
                
            if display:
                img_display = np.copy(img)
                
            hist_ini, bins_ini = im.gethist(img, histSize=histSize_fin, histRange=(0, 256))
            
            hist_new[0] -= (np.sum(hist_new) - np.sum(hist_ini))
            
            for n in range(bins_new.shape[0]-1):
                
                diff = int(hist_new[n] - hist_ini[n])
                
                if (diff<0): ### meaning pixels need to be shifted to higher intensities

                    idx, idy = np.where(img==bins_ini[n])
                    idn      = np.random.choice(np.asarray(range(len(idx))), abs(diff), replace=False)

                    n_=n+1
                    while (diff<0) and (n_ < bins_new.shape[0]):

                        diff_ = int(hist_new[n_] - hist_ini[n_])

                        if diff_>0:
                            ii = idn.shape[0] if diff_>idn.shape[0] else diff_
                            for i in range(ii):
                                img[idx[idn[i]],idy[idn[i]]] = bins_new[n_]
                            idn = idn[ii:]

                            hist_ini[n_] = hist_ini[n_] + ii
                            hist_ini[n]  = hist_ini[n]  - ii

                            diff = diff+ii

                        n_+=1

                elif (diff>0): ### meaning more pixels need to be shifted to lower intensities 

                    n_=n+1
                    while (diff>0) and (n_ < bins_new.shape[0]):

                        diff_ = int(hist_new[n_] - hist_ini[n_])

                        if diff_<0:

                            idx, idy = np.where(img==bins_ini[n_])
                            idn      = np.random.choice(np.asarray(range(len(idx))), abs(diff_), replace=False)

                            ii = idn.shape[0] if diff>idn.shape[0] else diff
                            for i in range(ii):
                                img[idx[idn[i]],idy[idn[i]]] = bins_new[n]
                            idn = idn[ii:]

                            hist_ini[n_] = hist_ini[n_] - ii
                            hist_ini[n]  = hist_ini[n]  + ii

                            diff = diff-ii

                        n_+=1
            
            if display:
                title_ = ["before", "after"]
                fig, ax = plt.subplots(2, 2, figsize = (16, 16))
                for n, item in enumerate([self.data[k], img]):
                    hist, bins = im.gethist(item)
                    ax[0,n].bar(bins, hist, width=1)
                    ax[0,n].set_title(title_[n], fontsize=18)
                    ax[1,n].imshow(im(item).adjust(lower=2, upper=99.8), cmap='magma')
                    ax[1,n].axis("off")
                fig.tight_layout()
                plt.show();
                
            self.data[k] = np.copy(img)

        return im.exit(self)
    
    def addnoise(self, noise_type='gauss', mean = 0, sigma = 1, s_vs_p = 0.5, amount = 0.004):

        """ The Function adds gaussian , salt-pepper , poisson and speckle noise in an image

        Parameters
        ----------
        image : ndarray
                Input image data. Will be converted to float.

        functions : 

                'gauss'     Gaussian-distributed additive noise.
                'poisson'   Poisson-distributed noise generated from the data.
                's&p'       Replaces random pixels with 0 or 1.
                'speckle'   Multiplicative noise using out = image + n*image,where
                            n is uniform noise with specified mean & variance.

        Ref
        ----------
        from: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

        """

        for n in range(self.N):

            if noise_type=='gauss':
                self.data[n] = self.data[n] + np.random.normal(mean,sigma,self.data[n].shape)

            elif noise_type=='sp':

                shape_ = self.data[n].shape
                size_  = self.data[n].size

                out = np.copy(self.data[n])

                # Salt mode
                num_salt = np.ceil(amount * size_ * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in shape_]
                out[coords] = 1

                # Pepper mode
                num_pepper = np.ceil(amount * size_ * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in shape_]
                out[coords] = 0

                self.data[n] = out

            elif noise_type=='poisson':
                vals = len(np.unique(self.data[n]))
                vals = 2 ** np.ceil(np.log2(vals))
                self.data[n] = np.random.poisson(self.data[n] * vals) / float(vals)

            elif noise_type=='speckle':
                row,col = self.data[n].shape
                self.data[n] = self.data[n] + self.data[n] * np.random.randn(row,col)

        return im.exit(self)


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):#dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def norm_minmse(gt, x, normalize_gt=True):
    """This function is adapted from Martin Weigert"""

    """
    normalizes and affinely scales an image pair such that the MSE is minimized  

    Parameters
    ----------
    gt: ndarray
        the ground truth image      
    x: ndarray
        the image that will be affinely scaled 
    normalize_gt: bool
        set to True of gt image should be normalized (default)
    Returns
    -------
    gt_scaled, x_scaled 
    """
    if normalize_gt:
        gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy = False)
    x = x.astype(np.float32, copy=False) - np.mean(x)
    #x = x - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    #gt = gt - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x

