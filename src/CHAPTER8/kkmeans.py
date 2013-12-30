#!/usr/bin/env python
#******************************************************************************
#  Name:     kkmeans.py
#  Purpose:  Perform kernel K-means clustering on multispectral imagery 
#  Usage:             
#    python kkmeans.py 
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
import auxil.header as header
from auxil.auxil import ctable
import os, time, sys
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte

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
    m = auxil.select_integer(1000,'Select training sample size')
    K = auxil.select_integer(6,'Select number of clusters')
    outfile, outfmt = auxil.select_outfilefmt()  
    if not outfile:
        return  
    kernel = auxil.select_integer(1,'Select kernel: 0=linear, 1=Gaussian')    
    print '========================='
    print '       kkmeans'
    print '========================='
    print 'infile:  '+infile
    print 'samples: '+str(m) 
    if kernel == 0:
        print 'kernel:  '+'linear' 
    else:
        print 'kernel:  '+'Gaussian'  
    start = time.time()                                     
#  input data matrix           
    XX = np.zeros((cols*rows,bands))      
    k = 0
    for b in pos:
        band = inDataset.GetRasterBand(b)
        band = band.ReadAsArray(x0,y0,cols,rows).astype(float)
        XX[:,k] = np.ravel(band)
        k += 1
#  training data matrix
    idx = np.fix(np.random.random(m)*(cols*rows)).astype(np.integer)
    X = XX[idx,:]  
    print 'kernel matrix...'
# uncentered kernel matrix    
    KK, gma = auxil.kernelMatrix(X,kernel=kernel)      
    if gma is not None:
        print 'gamma: '+str(round(gma,6))    
#  initial (random) class labels
    labels = np.random.randint(K,size = m)  
#  iteration
    change = True
    itr = 0
    onesm = np.mat(np.ones(m,dtype=float))
    while change and (itr < 100):
        change = False
        U = np.zeros((K,m))
        for i in range(m):
            U[labels[i],i] = 1
        M =  np.diag(1.0/(np.sum(U,axis=1)+1.0))
        MU = np.mat(np.dot(M,U))
        Z = (onesm.T)*np.diag(MU*KK*(MU.T)) - 2*KK*(MU.T)
        Z = np.array(Z) 
        labels1 = (np.argmin(Z,axis=1) % K).ravel()
        if np.sum(labels1 != labels):
            change = True
        labels = labels1   
        itr += 1
    print 'iterations: %i'%itr 
#  classify image
    print 'classifying...'
    i = 0
    A = np.diag(MU*KK*(MU.T))
    A = np.tile(A,(cols,1))
    class_image = np.zeros((rows,cols),dtype=np.byte)
    while i < rows:     
        XXi = XX[i*cols:(i+1)*cols,:]
        KKK,_ = auxil.kernelMatrix(X,XXi,gma=gma,kernel=kernel)
        Z = A - 2*(KKK.T)*(MU.T)
        Z= np.array(Z)
        labels = np.argmin(Z,axis=1).ravel()
        class_image[i,:] = (labels % K) +1
        i += 1   
    sys.stdout.write("\n")    
#  write to disk
    driver = gdal.GetDriverByName(outfmt)    
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Byte)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)               
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(class_image,0,0) 
    outBand.FlushCache() 
    outDataset = None
    inDataset = None
    if (outfmt == 'ENVI') and (K<19):
#  try to make an ENVI classification header file            
        hdr = header.Header() 
        headerfile = outfile+'.hdr'
        f = open(headerfile)
        line = f.readline()
        envihdr = ''
        while line:
            envihdr += line
            line = f.readline()
        f.close()         
        hdr.read(envihdr)
        hdr['file type'] ='ENVI Classification'
        hdr['classes'] = str(K)
        classlookup = '{0'
        for i in range(1,3*K):
            classlookup += ', '+str(str(ctable[i]))
        classlookup +='}'    
        hdr['class lookup'] = classlookup
        hdr['class names'] = [str(i+1) for i in range(K)]
        f = open(headerfile,'w')
        f.write(str(hdr))
        f.close()                 
    print 'result written to: '+outfile    
    print 'elapsed time: '+str(time.time()-start)                        
    print '--done------------------------'  
       
if __name__ == '__main__':
    main()    