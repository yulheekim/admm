import numpy as np
import cv2
import pybm3d
import skimage
import math

from constructGGt import constructGGt
from defGGt import fdown,upf
from proj import proj

def ADMM_super(y,h,K,lam,method,opts):
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % out = PlugPlayADMM_super(y,h,K,lambda,method,opts)
# %
# % inversion step: x=argmin_x(||Ax-y||^2+rho/2||x-(v-u)||^2)
# % denoising step: v=Denoise(x+u)
# %       update u: u=u+(x-v)
# %
# %Input:           y    -  the observed gray scale image
# %                 h    -  blur kernel
# %                 K    -  downsampling factor
# %              lambda  -  regularization parameter
# %              method  -  denoiser, e.g., 'BM3D'
# %       opts.rho       -  internal parameter of ADMM {1}
# %       opts.gamma     -  parameter for updating rho {1}
# %       opts.maxitr    -  maximum number of iterations for ADMM {20}
# %       opts.tol       -  tolerance level for residual {1e-4}
# %       ** default values of opts are given in {}.
# %
# %Output:          out  -  recovered gray scale image
# %
# %
# %Xiran Wang and Stanley Chan
# %Copyright 2016
# %Purdue University, West Lafayette, In, USA.
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Check inputs
    if y.all() and h.all() and K and lam and method:
        opts={}
    else:
        print('not enough input, try again \n')

    # % Check defaults
    if 'rho' not in opts:
        opts['rho'] = 1
    if 'max_itr' not in opts:
        opts['max_itr'] = 20
    if 'tol' not in opts:
        opts['tol'] = 1e-4
    if 'gamma' not in opts:
        opts['gamma']=1
    if 'print' not in opts:
        opts['print'] = False

    # % set parameters
    max_itr   = opts['max_itr']
    tol       = opts['tol']
    gamma     = opts['gamma']
    rho       = opts['rho']

    # %initialize variables
    rows_in = len(y)
    cols_in = len(y[0])
    rows      = rows_in*K
    cols      = cols_in*K
    N         = rows*cols

    # [G,Gt]    = defGGt(h,K) # functions for fdown(x, h, K) and upf(x, h, K)
    defGGt_h = h
    defGGt_K = K
    GGt       = constructGGt(h,K,rows,cols)
    # Gty       = Gt(y)
    Gty       = upf(y, defGGt_h, defGGt_K)
    # v         = imresize(y,K)
    v         = skimage.transform.resize(y, (np.shape(y)[0]*K,np.shape(y)[1]*K)) # can't find a good replacement here...
    
    x         = v
    u         = np.zeros(np.shape(v))
    residual  = float("inf")

    # %set function handle for denoiser
    # switch method
    #     case 'BM3D'
    #         denoise=@wrapper_BM3D
    #     case 'TV'
    #         denoise=@wrapper_TV
    #     case 'NLM'
    #         denoise=@wrapper_NLM
    #     case 'RF'
    #         denoise=@wrapper_RF
    #     otherwise
    #         error('unknown denoiser \n')
    # end

    # % main loop
    if opts['print']==True:
        print('Plug-and-Play ADMM --- Super Resolution \n')
        print('Denoiser = \n\n', method)
        print('itr \t ||x-xold|| \t ||v-vold|| \t ||u-uold|| \n')

    itr = 1
    while(residual>tol and itr<=max_itr):
        # %store x, v, u from previous iteration for psnr residual calculation
        x_old=x
        v_old=v
        u_old=u

        # %inversion step
        xtilde = v-u
        rhs = Gty + rho*xtilde
        x = (rhs - upf(np.fft.ifft2(np.fft.fft2(fdown(rhs, defGGt_h, defGGt_K)) / (GGt + rho)), defGGt_h, defGGt_K))/rho

        # %denoising step
        vtilde = x+u
        vtilde = proj(vtilde) # should be 512x512
        sigma  = math.sqrt(lam/rho)
        v      = pybm3d.bm3d.bm3d(vtilde,sigma) # BM3D(noisy_img, noise_std_dev)  source: https://github.com/ericmjonas/pybm3d


        # %update langrangian multiplier
        u      = u + (x-v)

        # %update rho
        rho=rho*gamma

        # %calculate residual
        residualx = (1/math.sqrt(N))*(math.sqrt(sum(sum((x-x_old)**2))))
        residualv = (1/math.sqrt(N))*(math.sqrt(sum(sum((v-v_old)**2))))
        residualu = (1/math.sqrt(N))*(math.sqrt(sum(sum((u-u_old)**2))))

        residual = residualx + residualv + residualu

        if opts['print']==True:
            print(itr, residualx, residualv, residualu)

        itr=itr+1
    return x
