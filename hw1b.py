from os import walk,listdir
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
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
    


def plot(c, D,  X_mn, ax):
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
    
   
    v=v.reshape((256,256))
    

   
   
    v=v+X_mn
    
   
    
    ax.imshow(v,cmap = cm.Greys_r)
    
   

if __name__ == '__main__':
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    N=sum([f.endswith('.tiff') for f in listdir('jaffe')])
    sze=256
    Ims=np.zeros((N,sze*sze))
    i=0
    for root, dirs, files in walk('jaffe'):
               
        for file in sorted(files):
          
            if (file and file.endswith('tiff')):
                im = Image.open(root+'/'+file).convert('L')
                Im=np.array(im).reshape(1,sze*sze)
                Ims[i]=Im
                i+=1
        

    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
   '''
    rng = np.random
    n_pcas=16
    
    #   Symbolic
    x = T.matrix("x")
    V=T.matrix('V')
    d = T.vector("d")
    lam=T.vector("lam")
    
    #   Initial D is a zero matrix
    D=np.zeros((sze*sze, n_pcas))
    
    #   Initial lambda_vector is a zero vectors
    lambda_vector=np.zeros((n_pcas,))
    
    
    #   Parameters
    n_iterations=1000
    eta = 0.01
    epsilon=0.001
   
   
    
    
    for i in range(0,n_pcas):
       
        t=1
        
        # Initial diff between d_new and d_old
        delta=100
        
        # Initial d_i
        d_shared=theano.shared(rng.randn(sze*sze), name="d")
        
        # X.d_i
        xd=T.dot(x,d_shared)
        
        # First term of the cost function
        cost_1=T.dot(T.transpose(xd),xd)
        
        # First term of the cost function : Norm(D^T.d_i*sqrt(lambda_vector))^2
        cost_2=((T.dot(T.transpose(V),d_shared)*T.sqrt(lam))**2).sum()
        
        
        cost=cost_1-cost_2
                
        
        
        #      Gradient of the cost
        gd=T.grad(-eta*cost,d_shared)

        # y        
        y=d_shared-gd    
        y=y/(T.sqrt((y**2).sum()))
        
        #   difference
        diff=T.sqrt(((d_shared-y)**2).sum())
    
        # Theano function that outputs the difference and the cost_1 and updates d_i
        grad_desc=theano.function(inputs=[x,lam,V], outputs=[diff,cost_1],updates=[(d_shared,y)],allow_input_downcast=True)
        
        #   While loop
        while((t<=n_iterations) and (delta>epsilon)):
            delta,lambda_i=grad_desc(X,lambda_vector,D)
        
        
        #   Return d_i in D
        D[:,i]=d_shared.get_value()
        
        #   Return lambda_i in lambda_vector
        lambda_vector[i]=lambda_i
        
            
   
    
    c = np.dot(D.T, X.T)
    for i in range(0,200,10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')

