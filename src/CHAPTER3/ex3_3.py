#!/usr/bin/env python
#Name:  ex3_3.py
import auxil.auxil as auxil
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

def main(): 
    gdal.AllRegister()
    infile = auxil.select_infile() 
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    else:
        return
#  spectral and spatial subsets    
    pos =  auxil.select_pos(bands)
    bands = len(pos)    
    x0,y0,rows,cols=auxil.select_dims([0,0,rows,cols])   
#  data matrix for difference images
    D = zeros((rows*cols,bands))
    i = 0                                   
    for b in pos:
        band = inDataset.GetRasterBand(b)
        tmp = band.ReadAsArray(x0,y0,cols,rows)\
                              .astype(float)
        D[:,i] = (tmp-(roll(tmp,1,axis=0)+\
                 roll(tmp,1,axis=1))/2).ravel()
        i += 1       
#  noise covariance matrix
    S_N = mat(D).T*mat(D)/(rows*cols-1)
    print 'Noise covariance matrix, file %s'%infile
    print S_N
    
if __name__ == '__main__':
    main()    