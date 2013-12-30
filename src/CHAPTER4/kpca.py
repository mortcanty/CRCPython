#!/usr/bin/env python
#******************************************************************************
#  Name:     kPCA.py
#  Purpose:  Perform kernel PCA on multispectral imagery 
#  Usage:             
#    python kpca.py 
#
#  Copyright (c) 2012, Mort Canty
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
import auxil.auxil as auxil
import sys, os, time
from numpy import *
from scipy import linalg
from scipy.cluster.vq import kmeans
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
import matplotlib.pyplot as plt
    
def main():
    gdal.AllRegister()
    path = auxil.select_directory('Choose working directory')
    if path:
        os.chdir(path)   
    infile = auxil.select_infile(title='Select an image') 
    if infile:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    else:
        return
    pos =  auxil.select_pos(bands) 
    if not pos:
        return   
    dims = auxil.select_dims([0,0,cols,rows])
    if dims:
        x0,y0,cols,rows = dims
    else:
        return   
    m = auxil.select_integer(2000,'Select sample size (0 for k-means)')

    n = auxil.select_integer(10,'Select number of eigenvalues')
    outfile, fmt = auxil.select_outfilefmt()  
    if not outfile:
        return  
    kernel = auxil.select_integer(1,'Select kernel: 0=linear, 1=Gaussian')    
    print '========================='
    print '       kPCA'
    print '========================='
    print 'infile:  '+infile
    print 'samples: '+str(m) 
    if kernel == 0:
        print 'kernel:  '+'linear' 
    else:
        print 'kernel:  '+'Gaussian'  
    start = time.time()                                     
    if kernel == 0:
        n = min(bands,n)
# construct data design matrices           
    XX = zeros((cols*rows,bands))      
    k = 0
    for b in pos:
        band = inDataset.GetRasterBand(b)
        band = band.ReadAsArray(x0,y0,cols,rows).astype(float)
        XX[:,k] = ravel(band)
        k += 1   
    if m > 0:   
        idx = fix(random.random(m)*(cols*rows)).astype(integer)
        X = XX[idx,:]  
    else:   
        print 'running k-means on 100 cluster centers...'
        X,_ = kmeans(XX,100,iter=1)
        m = 100
    print 'centered kernel matrix...'
# centered kernel matrix    
    K, gma = auxil.kernelMatrix(X,kernel=kernel)      
    meanK = sum(K)/(m*m)
    rowmeans = mat(sum(K,axis=0)/m)
    if gma is not None:
        print 'gamma: '+str(round(gma,6))    
    K = auxil.center(K)    
    print 'diagonalizing...'
# diagonalize
    try:
        w, v = linalg.eigh(K,eigvals=(m-n,m-1))
        idx = range(n)
        idx.reverse()  
        w = w[idx] 
        v = v[:,idx]    
#      variance of PCs        
        var = w/m
    except linalg.LinAlgError:
        print 'eigenvalue computation failed'
        sys.exit()   
#  dual variables (normalized eigenvectors)
    alpha = mat(v)*mat(diag(1/sqrt(w)))            
    print 'projecting...' 
#  projecting     
    image = zeros((rows,cols,n)) 
    for i in range(rows):
        XXi = XX[i*cols:(i+1)*cols,:]       
        KK,gma = auxil.kernelMatrix(X,XXi,kernel=kernel,gma=gma)
#  centering on training data: 
#      subtract column means
        colmeans = mat(sum(KK,axis=0)/m)
        onesm = mat(ones(m))
        KK = KK - onesm.T*colmeans
#      subtract row means
        onesc = mat(ones(cols))                
        KK = KK - rowmeans.T*onesc
#      add overall mean
        KK = KK + meanK 
#      project
        image[i,:,:] = KK.T*alpha    
#  write to disk
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,n,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)        
    for k in range(n):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(image[:,:,k],0,0) 
        outBand.FlushCache() 
    outDataset = None
    inDataset = None
    print 'result written to: '+outfile    
    print 'elapsed time: '+str(time.time()-start)                        
    plt.plot(range(1,n+1), var,'k-')
    plt.title('kernel PCA') 
    plt.xlabel('principal component')
    plt.ylabel('Variance')        
    plt.show()        
    print '--done------------------------'  
       
if __name__ == '__main__':
    main()    