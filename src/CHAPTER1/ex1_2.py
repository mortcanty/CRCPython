#!/usr/bin/env python
#  Name:     ex1_2.py
import auxil.auxil as auxil
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import matplotlib.pyplot as plt

def main(): 
    gdal.AllRegister()
    infile = auxil.select_infile(filt='*.xml') 
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    else:
        return                               
    pos =  auxil.select_pos(bands) 
    x0,y0,rows,cols=auxil.select_dims([0,0,rows,cols])
#  BSQ array
    image = zeros((len(pos),rows,cols),dtype=complex64) 
    k = 0                                   
    for b in pos:
        band = inDataset.GetRasterBand(b)
        image[k,:,:]=band.ReadAsArray(x0,y0,cols,rows)\
                                       .astype(complex)
        k += 1
    inDataset = None
#  display magnitude in linear 2% stretch    
    band0 = abs(image[0,:,:]) 
    band0 = auxil.lin2pcstr(band0)
    mn = amin(band0)
    mx = amax(band0)
    plt.imshow((band0-mn)/(mx-mn), cmap='gray') 
    plt.show()                           

if __name__ == '__main__':
    main()    