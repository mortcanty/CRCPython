#!/usr/bin/env python
#Name:  ex5_2.py
import auxil.auxil as auxil
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GDT_Byte
import cv2 as cv 
  
def main(): 
    gdal.AllRegister()
    infile = auxil.select_infile() 
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
    else:
        return   
#  read first image band
    rasterBand = inDataset.GetRasterBand(1)
    band = rasterBand.ReadAsArray(0,0,cols,rows)\
                                  .astype(uint8)       
#  edge detection, window size 7x7
    result = cv.Canny(band,50,150)     
#  write to disk       
    outfile,fmt = auxil.select_outfilefmt() 
    if outfile:
        driver = gdal.GetDriverByName(fmt)   
        outDataset = driver.Create(outfile,
                        cols,rows,1,GDT_Byte)         
        outBand = outDataset.GetRasterBand(1)
        outBand.WriteArray(result,0,0) 
        outBand.FlushCache() 
        outDataset = None    
    inDataset = None        
 
if __name__ == '__main__':
    main()    