from os import walk,listdir
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs,neibs2images

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs, n_blocks):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
           
            cij = c[:nc, n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
            Dij = D[:, :nc]
            plot(cij, Dij, n_blocks, X_mn, axarr[i, j])

    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)

def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    f, axarr = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            n=i*4+j
            # Pick the nth eigenvector from D
            pc=D[:,n].T
            pc=np.reshape(pc,(sz,sz))
            axarr[i,j].imshow(pc,cmap = cm.Greys_r)
          
    f.savefig(imname.format(sz, sz))
    
    plt.close(f)
    


def plot(c, D, n_blocks, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
   
   
    v=np.dot(c.T,D.T)
 
   
    images = T.tensor4('images')
    

    sz=int(256/n_blocks)
    
    #   Unflatten    
    neibs = images2neibs(images, neib_shape=(sz, sz))   
    im_new = neibs2images(neibs, (sz, sz), (256, 256))
    inv_window = theano.function([neibs], im_new,allow_input_downcast=True)
    vv = inv_window(v)
    
    
    X_mn=X_mn.reshape(1,1,sz,sz).repeat(n_blocks, 0)
    X_mn=X_mn.repeat(n_blocks, 1)
    X_mn=X_mn.swapaxes(1, 2).reshape(-1,sz*n_blocks)
   
    vv=vv+X_mn
    
    
    
    ax.imshow(vv,cmap = cm.Greys_r)
    
   


def main():
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    
    N=sum([f.endswith('.tiff') for f in listdir('jaffe')])
    sze=256
    Ims=np.zeros((N,sze,sze))
    i=0
    for root, dirs, files in walk('jaffe'):
               
        for file in sorted(files):
          
            if (file and file.endswith('tiff')):
                im = Image.open(root+'/'+file).convert('L')
                
                Im=np.array(im)
                Ims[i]=Im
                i+=1
        

    szs = [16, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in  zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''
        n_blocks_in_image=int((256*256)/(sz*sz))
        X=np.zeros((N*n_blocks_in_image,sz*sz))
        
        images = T.tensor4('images')
        neibs = images2neibs(images, neib_shape=(sz, sz))
        window_function = theano.function([images], neibs,allow_input_downcast=True)
        for k in range(0,N):
            im_val = Ims[k].reshape(( 1,1, 256, 256))
            X[k*n_blocks_in_image:(k+1)*n_blocks_in_image,:] = window_function(im_val)
        
        
        
        
        

        X_mn = np.mean(X, 0)
        X = X - np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        w,D =np.linalg.eigh(np.dot(X.T,X))        
        idx = w.argsort()[::-1]   
        w = w[idx]
        D = D[:,idx]
        
        
        c = np.dot(D.T, X.T)

        for i in range(0, 200, 10):
            plot_mul(c, D, i, X_mn.reshape((sz, sz)),num_coeffs=nc, n_blocks=int(256/sz))

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
