#!/usr/bin/env python
#  Name:     ex3_1.py
import auxil.auxil as auxil
from numpy import *
from numpy import fft
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import matplotlib.pyplot as plt

def main():        
    gdal.AllRegister()
    infile = auxil.select_infile() 
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
    else:
        return
    band = inDataset.GetRasterBand(1)  
    image = band.ReadAsArray(0,0,cols,rows) \
                               .astype(float)
#  arrays of i and j values    
    a = reshape(range(rows*cols),(rows,cols))
    i = a % cols
    j = a / cols
#  shift Fourier transform to center    
    image = (-1)**(i+j)*image
#  compute power spectrum and display    
    image = log(abs(fft.fft2(image))**2)
    mn = amin(image)
    mx = amax(image)
    plt.imshow((image-mn)/(mx-mn), cmap='gray' )  
    plt.show()                        

if __name__ == '__main__':
    main()    