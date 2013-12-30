#!/usr/bin/env python
#Name:  ex8_1.py
import auxil.auxil as auxil
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
from scipy.cluster.vq import kmeans,vq

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
    pos =  auxil.select_pos(bands)
    bands = len(pos)    
    x0,y0,rows,cols=auxil.select_dims([0,0,rows,cols])   
    K = auxil.select_integer(6,msg='Number clusters')        
    G = zeros((rows*cols,len(pos))) 
    k = 0                                   
    for b in pos:
        band = inDataset.GetRasterBand(b)
        G[:,k] = band.ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()
        k += 1        
    centers, _ = kmeans(G,K)
    labels, _ = vq(G,centers)      
    outfile,fmt = auxil.select_outfilefmt() 
    if outfile:
        driver = gdal.GetDriverByName(fmt)   
        outDataset = driver.Create(outfile,
                        cols,rows,1,GDT_Byte)         
        outBand = outDataset.GetRasterBand(1)
        outBand.WriteArray(reshape(labels,(rows,cols))\
                                              ,0,0) 
        outBand.FlushCache() 
        outDataset = None    
    inDataset = None        
 
if __name__ == '__main__':
    main()    