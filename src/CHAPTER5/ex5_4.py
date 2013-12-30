#!/usr/bin/env python
#Name:  ex5_4.py
import auxil.auxil as auxil
import os
from numpy import *
from osgeo import gdal
import scipy.ndimage.interpolation as ndii
from osgeo.gdalconst import GA_ReadOnly, GDT_CFloat32
  
def main(): 
    gdal.AllRegister()
    path = auxil.select_directory('Working directory')
    if path:
        os.chdir(path)        
    file0=auxil.select_infile(title='Base image') 
    if file0:                   
        inDataset0 = gdal.Open(file0,GA_ReadOnly)     
        cols0 = inDataset0.RasterXSize
        rows0 = inDataset0.RasterYSize
        print 'Base image: %s'%file0    
    else:
        return     
    rasterBand = inDataset0.GetRasterBand(1)
    span0 = rasterBand.ReadAsArray(0,0,cols0,rows0)
    rasterBand = inDataset0.GetRasterBand(4)
    span0 += 2*rasterBand.ReadAsArray(0,0,cols0,rows0)
    rasterBand = inDataset0.GetRasterBand(6)
    span0 += rasterBand.ReadAsArray(0,0,cols0,rows0)  
    span0 = log(real(span0))      
    inDataset0 = None   
    file1=auxil.select_infile(title='Warp image') 
    if file1:                  
        inDataset1 = gdal.Open(file1,GA_ReadOnly)     
        cols1 = inDataset1.RasterXSize
        rows1 = inDataset1.RasterYSize
        print 'Warp image: %s'%file1    
    else:
        return   
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return   
    image1 = zeros((6,rows1,cols1),dtype=cfloat)                                   
    for k in range(6):
        band = inDataset1.GetRasterBand(k+1)
        image1[k,:,:]=band\
          .ReadAsArray(0,0,cols1,rows1).astype(cfloat)    
    inDataset1 = None 
    span1 = sum(image1[[0,3,5] ,:,:],axis=0)\
                                        +image1[3,:,:]                   
    span1 = log(real(span1))                
    scale,angle,shift = auxil.similarity(span0, span1)    
    tmp_real = zeros((6,rows0,cols0))
    tmp_imag = zeros((6,rows0,cols0))
    for k in range(6): 
        bn1 = real(image1[k,:,:])                   
        bn2 = ndii.zoom(bn1, 1.0/scale)
        bn2 = ndii.rotate(bn2, angle)
        bn2 = ndii.shift(bn2, shift)
        tmp_real[k,:,:] = bn2[0:rows0,0:cols0] 
        bn1 = imag(image1[k,:,:])                   
        bn2 = ndii.zoom(bn1, 1.0/scale)
        bn2 = ndii.rotate(bn2, angle)
        bn2 = ndii.shift(bn2, shift)
        tmp_imag[k,:,:] = bn2[0:rows0,0:cols0] 
    image2 = tmp_real + 1j*tmp_imag                  
    driver = gdal.GetDriverByName(fmt)   
    outDataset = driver.Create(outfile,
                    cols0,rows0,6,GDT_CFloat32)
    for k in range(6):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(image2[k,:,:],0,0) 
        outBand.FlushCache()
    outDataset = None
    print 'Warped image written to: %s'%outfile        

if __name__ == '__main__':
    main()    