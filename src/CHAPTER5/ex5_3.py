#!/usr/bin/env python
#Name:  ex5_3.py
import auxil.auxil as auxil
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import cv2 as cv 
import matplotlib.pyplot as plt
  
def main(): 
    gdal.AllRegister()
#  read first band of an MS image   
    infile = auxil.select_infile() 
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize
    else:
        return   
    rasterBand = inDataset.GetRasterBand(1) 
    band = rasterBand.ReadAsArray(0,0,cols,rows)                              
#  find and display contours    
    edges = cv.Canny(band, 20, 80)    
    contours,hierarchy = cv.findContours(edges,\
             cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    arr = zeros((rows,cols),dtype=uint8)
    cv.drawContours(arr, contours, -1, 255)
    plt.imshow(arr,cmap='gray') ;  plt.show()
#  determine Hu moments        
    num_contours = len(hierarchy[0])    
    hus = zeros((num_contours,7),dtype=float32)
    for i in range(num_contours): 
        arr = arr*0  
        cv.drawContours(arr, contours, i, 1)                      
        m = cv.moments(arr)
        hus[i,:] = cv.HuMoments(m).ravel()
#  plot histograms of logarithms of the Hu moments        
    for i in range(7): 
        idx = where(hus[:,i]>0)  
        hist,_ = histogram(log(hus[idx,i]),50)    
        plt.plot(range(50), hist, 'k-'); plt.show()        

if __name__ == '__main__':
    main()    