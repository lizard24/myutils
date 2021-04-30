import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('axes',edgecolor='#AEB6BF')
from myutils.myimageprocess import im
from twilio.rest import Client
from read_roi import read_roi_file as read_roi
from scipy import ndimage


def extract_profile(image, x0, y0, x1, y1, width=1, num=None):
    """
    Extract a line profile from (x0, y0) to (x1, y1).
    
    Parameters
    ----------
    image :  2D ndarray
        the image to extract from
    x0, y0 : int
        coordinates of start of profile
    x1, y1 : int
        coordinates of profile end
    width : int
        profile width in pixels (number of parallel profiles to average over)
    Returns
    -------
    
    a line profile of the image, interpolated to match the pixel spacing
    """
    w = int(np.floor(0.5 * width))
    
    Dx = x1 - x0
    Dy = y1 - y0
    
    l = np.sqrt((Dx ** 2 + Dy ** 2)) if num is None else num
    
    dx = Dx / l
    dy = Dy / l
    
    if Dx == 0 and Dy == 0: #special case - profile is orthogonal to current plane
        d_x = w
        d_y = w
    else:
        d_x = w * abs(dy)
        d_y = w * abs(dx)
    
    #pixel indices at 1-pixel spacing
    t = np.arange(np.ceil(l))
    
    x_0 = min(x0, x1)
    y_0 = min(y0, y1)
    
    d__x = abs(d_x) + 1
    d__y = abs(d_y) + 1
    
    ims = image[int(min(x0, x1) - d__x):int(max(x0, x1) + d__x + 1),
                int(min(y0, y1) - d__y):int(max(y0, y1) + d__y + 1)].squeeze()
    
    splf = ndimage.spline_filter(ims)
    
    p = np.zeros(len(t))
    
    x_c = t * dx + x0 - x_0
    y_c = t * dy + y0 - y_0
    

    for i in range(-w, w + 1):
        p += ndimage.map_coordinates(splf, np.vstack([x_c + d__x + i * dy, y_c + d__y - i * dx]),
                                     prefilter=False)
        
    p = p / (2 * w + 1)
    
    return p


def _roi(img, file=None, num=None, width=1):

    """
    Extracts either cropped ROI or Lineprofile from image.
    
    Parameters
    ----------
    image :  2D ndarray
        the image to extract from
    file  :  str
        '.roi' file of lineprofile or ROI
    num  : int
        optional: for line profile; length of line profile; default: no interpolation
    width : int
        optional: for line profile; profile width in pixels (number of parallel profiles to average over)
    Returns
    -------
    
    cropped out ROI (=image) or line profile (1d array)
    """

    if not file is None:
   
        roi = read_roi(file)[file.split('/')[-1].split('.roi')[0]]
    
        try:
            l, t, w, h = roi['left'], roi['top'], roi['width'], roi['height']
            l, t, w, h = map(lambda x: int(x), (l, t, w, h))

            dims_first, dims_last = img.shape[:-2], img.shape[-2:]
            N = np.prod(dims_first) if not dims_first == () else 1
            img = np.reshape(img, (N,)+dims_last)

            img = img[:,t:t+h+1, l:l+w+1]

            dims_last = img.shape[-2:]
            img = np.reshape(img, dims_first+dims_last)

        except:
            x1, x2, y1, y2 = roi['x1'], roi['x2'], roi['y1'], roi['y2']
            x1, x2, y1, y2 = map(lambda x: int(x), (x1, x2, y1, y2))
                        
            img = extract_profile(img, y1, x1, y2, x2, width=width, num=num)
            
            #x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)
            #img = ndimage.map_coordinates(img, np.vstack((y, x)))
            
    return img


def myfigure( img,
              header_left = None,
              header_top  = None,
	      title       = None,
              xlim    = None,
              ylim    = None,
              pmin    = None,
              pmax    = None,
              rot     = False,
              save    = False,
              display = True,
	      roi     = None ):

    if save:
        generate_folder(save, level=-1)
    
    ### img is list of numpy arrays
    img = np.asarray(img)
    
    if len(img.shape)==3:
        img = np.reshape(img, (img.shape[0],1,)+img.shape[1:])        
    if rot:
        img = np.swapaxes(img, 0, 1)

    img = _roi(img, file=roi)
    
    if not (pmin, pmax) == (None, None):
        if (pmin!=None) and (pmax==None):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img[i,j] = img[i,j] - np.percentile(img[i,j],pmin)
        elif (pmin==None) and (pmax!=None):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img[i,j] = img[i,j] / np.percentile(img[i,j],pmax)
        else:
            img = im(img).adjust(lower=pmin, upper=pmax)
    
    (fig_cols, fig_rows) = img.shape[:2]
    
    interval = 18/fig_cols
    fig, ax = plt.subplots(fig_rows, fig_cols, figsize = (interval*fig_cols, interval*fig_rows))
    ax = np.reshape(ax, (fig_rows, fig_cols))

    if not title is None:
        fig.suptitle(title, fontsize=int(200/fig_cols))
                           
    for row in range(fig_rows):
        for col in range(fig_cols):
            
            ax[row,col].imshow(img[col,row], cmap='magma')
            ax[row,col].get_xaxis().set_ticks([])
            ax[row,col].get_yaxis().set_ticks([])
            if not xlim is None:
                ax[row,col].set_xlim(xlim[0], xlim[1])
            if not ylim is None:
                ax[row,col].set_ylim(ylim[0], ylim[1])
            if header_left and (col==0):
                try:
                    ax[row,col].set_ylabel(header_left[row], FontSize=int(150/fig_cols))
                except:
                    pass
            if header_top and (row==0):
                try:
                    ax[row,col].set_title(header_top[col], FontSize=int(150/fig_cols))
                except:
                    pass
            
    fig.tight_layout()
    if save:
        plt.savefig(save+'.png')
    if display:
        plt.show(fig)
    else:
        plt.close(fig)
        
        

def generate_folder(folder, level=0):
    
    strings = folder.split('/')
    strings = [item for item in strings if item!='']
    
    if (level>0):
        print("Wrong value for level in generate_folder. Return without action.")
        return
    
    if len(strings) <= abs(level):
        return
    
    for n in range(len(strings)+level):
        new_folder = '/'.join(strings[:n+1])
        if not os.path.exists(new_folder): os.makedirs(new_folder)
        
        
def copy_file( source_file      = None,
               target_file      = None,
               source_dir       = None,
               target_dir       = None,
               overwrite_target = False,
               verbose          = True ):
    
        """ Moves source_file from source_dir to target_dir and renames it to target_file if specified.
        
        Q&A
        ----------
        Does this function overwrite file with same name in 'target_dir'? Not in default (see 'overwrite_target').
        Does this function create 'target_dir' if it doesn't exist? Yes.
        
        Parameters
        ----------
        source_file      : str; for example: 'data.txt'
        target_file      : str; for example: 'data.txt'
                           if set to 'None': target_file=source_file
        source_dir       : str; for example: 'data/source' 
        target_dir       : str; for example: 'data/target'
        overwrite_target : boolean; if set to 'True': function overwrites file in 'target_dir' with same name as 'target_file'
        verbose          : boolean; if set to 'True': function informs user if file is overwritten in 'target_dir'
        
        to do
        ----------
        allow folder structure in 'source_file' and 'target_file'
        
        """
        
        if target_file is None: target_file = source_file

        while source_dir[0]  == '/': source_dir=source_dir[1:]
        while source_dir[-1] == '/': source_dir=source_dir[:-1]
        while target_dir[0]  == '/': target_dir=target_dir[1:]
        while target_dir[-1] == '/': target_dir=target_dir[:-1]

        source_dir  = source_dir.replace('/','\\')
        target_dir  = target_dir.replace('/','\\')
        
        ### return if file already exists
        if (source_dir==target_dir) and (source_file==target_file):
            return
        
        ### return of 'source_file' and/or 'target_file' contain folder structure
        if ('/' in source_file) or ('\\' in source_file) or ('/' in target_file) or ('\\' in target_file):
            print("Return without execution: 'source_file' and/or 'target_file' contain forbidden characters.")
            return

        ### return if source file doesn't exist
        if not os.path.exists('%s/%s' % (source_dir, source_file)):
            print("Return without execution: Source file doesn't exist.")
            return
        
        ### return if file with target_file already exists in target_dir and overwrite_target is set to False 
        if (os.path.exists('%s/%s' % (target_dir, target_file))) and (overwrite_target is False):
            print("Return without execution: 'target_file' exists in 'target_dir' and 'overwrite_target' is set to 'False'.")
            return
        
        
        ### temporarily changes file in target_dir that has the same name as source_file
        ### -> only if target_file!=source_file.
        ### Note: we have already checked above if overwriting is ok
        placeholder = True if (os.path.exists('%s/%s' % (target_dir, source_file))) and (source_file != target_file) else False
        
        if (verbose==True) and (os.path.exists('%s/%s' % (target_dir, target_file))):
            print("File '%s\\%s' is overwritten." % (target_dir, target_file))
        
        if placeholder:
            placeholder_file = 'placeholder_'+source_file
            if not os.path.exists( '%s/%s' % (target_dir, placeholder_file) ):
                os.rename  ( os.path.join( target_dir, source_file ) , os.path.join( target_dir, placeholder_file ) )
            else:
                print("Return without execution: '%s' already exists in 'target_dir'. Please tidy up 'target_dir' and try again." % placeholder_file)
                return
            
        generate_folder(target_dir)
        
        shutil.copy( os.path.join( source_dir, source_file ) , target_dir)
        if source_file!=target_file:
            if os.path.exists('%s/%s' % (target_dir, target_file)):
                os.remove('%s/%s' % (target_dir, target_file))
            os.rename  ( os.path.join( target_dir, source_file ) , os.path.join( target_dir, target_file ) )
            
        if placeholder:
            os.rename  ( os.path.join( target_dir, placeholder_file ) , os.path.join( target_dir, source_file ) )
                
def whatsapp(body="Done!"):
    
    account_sid = 'ACa0b548abba61c5189a49f6dc1ec3deb5'   ### os.environ['TWILIO_ACCOUNT_SID']
    auth_token  = '395fd1e030b2fb6a200e96cae1f385f1'     ### os.environ['TWILIO_AUTH_TOKEN']
    client      = Client(account_sid, auth_token)

    message = client.messages.create(body  = body,
                                     from_ = 'whatsapp:+14155238886',
                                     to    = 'whatsapp:+447878279887')
    
