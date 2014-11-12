#!/usr/bin/env python
#  Name:  
#    registerSAR.py
#
#  Purpose:  
#    Perfrom image-image registration of two polarimetric SAR images 
#    via similarity warping. Assumes 9-bnad quad pol, 4-band dual pol
#    or one-band single pol SAR images. Warp image must completely
#    overlap base image.
#
#  Usage:             
#    import registerSAR
#    python registerSAR.py
#
#  Copyright (c) 2014, Mort Canty
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
import os, time
import numpy as np
from osgeo import gdal
import scipy.ndimage.interpolation as ndii
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
  
def registerSAR(file0, file1, outfile, fmt): 
    start = time.time()   
    gdal.AllRegister()
    inDataset0 = gdal.Open(file0, GA_ReadOnly)     
    cols = inDataset0.RasterXSize
    rows = inDataset0.RasterYSize
    bands = inDataset0.RasterCount
    print 'Base image: %s' % file0                   
    inDataset1 = gdal.Open(file1, GA_ReadOnly)     
    cols1 = inDataset1.RasterXSize
    rows1 = inDataset1.RasterYSize
    bands1 = inDataset1.RasterCount
    print 'Warp image: %s' % file1      
    if  bands != bands1:
        print 'Number of bands must be equal'
        return 0
    rasterBand = inDataset0.GetRasterBand(1)
    span0 = rasterBand.ReadAsArray(0, 0, cols, rows)
    if bands == 9:
        print 'warping 9 bands (quad pol)...'
        image2 = np.zeros((9, rows, cols))
        rasterBand = inDataset0.GetRasterBand(6)
        span0 += rasterBand.ReadAsArray(0, 0, cols, rows)
        rasterBand = inDataset0.GetRasterBand(9)
        span0 += rasterBand.ReadAsArray(0, 0, cols, rows)  
        span0 = np.log(np.nan_to_num(span0)+0.001)
        image1 = np.zeros((9, rows1 + 100, cols1 + 100), dtype=np.float32)                                   
        for k in range(9):
            band = inDataset1.GetRasterBand(k + 1)
            image1[k, 0:rows1, 0:cols1] = band.ReadAsArray(0, 0, cols1, rows1).astype(np.float32)    
        span1 = np.sum(image1[[0, 5, 8] , :, :], axis=0)                   
        span1 = np.log(np.nan_to_num(span1) + 0.001)                
        scale, angle, shift = auxil.similarity(span0, span1)   
#      warp image 
        for k in range(9): 
            bn1 = np.nan_to_num(image1[k, :, :])                  
            bn2 = ndii.zoom(bn1, 1.0 / scale)
            bn2 = ndii.rotate(bn2, angle)
            bn2 = ndii.shift(bn2, shift)
            image2[k, :, :] = bn2[0:rows, 0:cols] 
    elif bands == 4:
        print 'warping 4 bands (dual pol)...'
        image2 = np.zeros((4, rows, cols))
        rasterBand = inDataset0.GetRasterBand(4)
        span0 += rasterBand.ReadAsArray(0, 0, cols, rows)
        span0 = np.log(np.nan_to_num(span0)+0.001)      
        image1 = np.zeros((4, rows1 + 100, cols1 + 100), dtype=np.float32)                                   
        for k in range(4):
            band = inDataset1.GetRasterBand(k + 1)
            image1[k, 0:rows1, 0:cols1] = band.ReadAsArray(0, 0, cols1, rows1).astype(np.float32)  
        span1 = np.sum(image1[[0, 3] , :, :], axis=0)          
        span1 = np.log(np.nan_to_num(span1) + 0.001)          
        scale, angle, shift = auxil.similarity(span0, span1)    
        for k in range(4): 
            bn1 = np.nan_to_num(image1[k, :, :])                  
            bn2 = ndii.zoom(bn1, 1.0 / scale)
            bn2 = ndii.rotate(bn2, angle)
            bn2 = ndii.shift(bn2, shift)
            image2[k, :, :] = bn2[0:rows, 0:cols] 
    elif bands == 1:
        print 'warping 1 band (single pol)...'
        image2 = np.zeros((1, rows, cols))
        band = inDataset1.GetRasterBand(1)
        image1 = np.zeros((rows1 + 100, cols1 + 100),dtype=np.float32)
        image1[0:rows1, 0:cols1] = band.ReadAsArray(0, 0, cols1, rows1).astype(np.float32)
        span0 = np.log(np.nan_to_num(span0) + 0.001)
        span1 = np.log(np.nan_to_num(image1) + 0.001)   
        scale, angle, shift = auxil.similarity(span0, span1) 
        bn1 = np.nan_to_num(image1)                 
        bn2 = ndii.zoom(bn1, 1.0 / scale)
        bn2 = ndii.rotate(bn2, angle)
        bn2 = ndii.shift(bn2, shift)
        image2[0, :, :] = bn2[0:rows, 0:cols]        
    driver = gdal.GetDriverByName(fmt)   
    outDataset = driver.Create(outfile,cols, rows, bands, GDT_Float32)
    geotransform = inDataset0.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    projection = inDataset0.GetProjection()        
    if projection is not None:
        outDataset.SetProjection(projection) 
    for k in range(bands):        
        outBand = outDataset.GetRasterBand(k + 1)
        outBand.WriteArray(image2[k, :, :], 0, 0) 
        outBand.FlushCache()
    outDataset = None
    print 'Warped image written to: %s' % outfile
    print 'elapsed time: ' + str(time.time() - start)   
    return 1     
    
def main(): 
    print '========================='
    print '     Register SAR'
    print '========================='
    print time.asctime()  
    path = auxil.select_directory('Working directory')
    if path:
        os.chdir(path)        
    file0=auxil.select_infile(title='Base image') 
    if not file0:                   
        return  
    file1=auxil.select_infile(title='Warp image') 
    if not file1:                  
        return       
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return  
    if registerSAR(file0,file1,outfile,fmt):
        print 'done' 
    else:
        print 'registerSAR failed'        

if __name__ == '__main__':
    main()    
