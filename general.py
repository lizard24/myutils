import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('axes',edgecolor='#AEB6BF')
from myimageprocess import im

def myfigure( img,
              header_left = None,
              header_top  = None,
              xlim    = None,
              ylim    = None,
              pmin    = None,
              pmax    = None,
              rot     = False,
              save    = False,
              display = True ):
    
    ### img is list of numpy arrays
    img = np.asarray(img)
    
    if len(img.shape)==3:
        img = np.reshape(img, (img.shape[0],1,)+img.shape[1:])        
    if rot:
        img = np.swapaxes(img, 0, 1)
    
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
    
    from twilio.rest import Client
    
    account_sid = 'ACa0b548abba61c5189a49f6dc1ec3deb5'   ### os.environ['TWILIO_ACCOUNT_SID']
    auth_token  = '395fd1e030b2fb6a200e96cae1f385f1'     ### os.environ['TWILIO_AUTH_TOKEN']
    client      = Client(account_sid, auth_token)

    message = client.messages.create(body  = body,
                                     from_ = 'whatsapp:+14155238886',
                                     to    = 'whatsapp:+447878279887')
    
