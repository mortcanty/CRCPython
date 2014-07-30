#!/usr/bin/env python
#******************************************************************************
#  Name:     cloudpca.py
#  Purpose:  Illustrate parallel PCA with several multispectral images on Picloud
#  Usage:             
#    python cloudpca.py infilenames
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

import sys,os,auxil,tempfile,cloud   
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import matplotlib.pyplot as plt
    
def pca(fname):
# place image on local file system    
    img = tempfile.NamedTemporaryFile()
    cloud.files.get(fname,img.name)
# read image    
    gdal.AllRegister()
    inDataset = gdal.Open(img.name,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize    
    bands = inDataset.RasterCount                             
# construct data design matrices           
    X = zeros((rows*cols,bands),dtype=float32)      
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        x = band.ReadAsArray(0,0,rows,cols).astype(float32).ravel()
        X[:,b] = x - mean(x)     
    inDataset = None          
    X = mat(X)
    C = X.T*X/(cols*rows-1)     
    lams,U = linalg.eigh(C)
    idx = argsort(lams)[::-1]
    U = U[:,idx]                 
    return reshape(array(X*U),(rows,cols,bands))   
              
def main():
# ------------test---------------------------------------------------------------    
    if len(sys.argv) == 1:        
        (sys.argv).extend(['d:\\imagery\\aster1.tif','d:\\imagery\\aster2.tif'])
# -------------------------------------------------------------------------------    
    cloud.setkey(2329,'270cb3cccb9beb65d2f424b24ccbd5a920c5ccef')                  
    print '========================='
    print '       PCA on Picloud'
    print '========================='
    fnames = sys.argv[1:]
    print 'infiles:  '+ str(fnames)    
# upload to picloud     
    cnames = []    
    for fname in fnames:
        cloud.files.put(fname)
        cnames.append(os.path.basename(fname))
# process in parallel        
    jids = cloud.map(pca,cnames)    
# get the result    
    result = cloud.result(jids)
# delete picloud files    
    for f in cloud.files.list():
        cloud.files.delete(f)
# display        
    for PCs in result:
        rgb = PCs[:,:,0:3]  
        for i in range(3):
            rgb[:,:,i] = auxil.histeqstr(rgb[:,:,i])           
        mn = amin(rgb)
        mx = amax(rgb)
        rgb = (rgb-mn)/(mx-mn)
        plt.imshow(rgb) 
        plt.show()                              
    print '--done------------------------'  
       
if __name__ == '__main__':
    main()    