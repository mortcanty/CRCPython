#!/usr/bin/env python
#Name:  ex3_2.py
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
#  transposed data matrix
    m = rows*cols
    G = zeros((bands,m))                                  
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        tmp = band.ReadAsArray(0,0,cols,rows)\
                              .astype(float).ravel()
        G[b,:] = tmp - mean(tmp) 
    G = mat(G)           
#  covariance matrix
    S = G*G.T/(m-1)   
#  diagonalize and sort eigenvectors  
    lamda,W = linalg.eigh(S)
    idx = argsort(lamda)[::-1]
    lamda = lamda[idx]
    W = W[:,idx]                    
#  get principal components and reconstruct
    r = 3
    Y = W.T*G    
    G_r = W[:,:r]*Y[:r,:]
#  reconstruction error covariance matrix
    print  (G-G_r)*(G-G_r).T/(m-1) 
#  Equation (3.45)       
    print  W[:,r:]*diag(lamda[r:])*W[:,r:].T                       
    inDataset = None        
 
if __name__ == '__main__':
    main()    