#!/usr/bin/env python
#******************************************************************************
#  Name:     atwt.py
#  Purpose:  Panchromatic sharpening with the a trous wavelet transform
 
#  Usage:             
#    python atwt.py 
#
#  Copyright (c) 2013, Mort Canty
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
import scipy.ndimage.interpolation as ndii
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32      

def main():
    gdal.AllRegister()
    path = auxil.select_directory('Choose working directory')
    if path:
        os.chdir(path)        
#  MS image    
    file1 = auxil.select_infile(title='Choose MS image') 
    if file1:                   
        inDataset1 = gdal.Open(file1,GA_ReadOnly)     
        cols = inDataset1.RasterXSize
        rows = inDataset1.RasterYSize    
        bands = inDataset1.RasterCount
    else:
        return
    pos1 =  auxil.select_pos(bands) 
    if not pos1:
        return   
    num_bands = len(pos1)
    dims = auxil.select_dims([0,0,cols,rows])
    if dims:
        x10,y10,cols1,rows1 = dims
    else:
        return 
#  PAN image     
    file2 = auxil.select_infile(title='Choose PAN image') 
    if file2:                  
        inDataset2 = gdal.Open(file2,GA_ReadOnly)     
        cols = inDataset2.RasterXSize
        rows = inDataset2.RasterYSize    
        bands = inDataset2.RasterCount
    else:
        return   
    if bands>1:
        print 'Must be a single band (panchromatic) image'
        return 
    dims=auxil.select_dims([0,0,cols,rows])  
    if dims:
        x20,y20,cols2,rows2 = dims
    else:
        return 
#  outfile
    outfile, fmt = auxil.select_outfilefmt()  
    if not outfile:
        return 
#  resolution ratio      
    ratio = auxil.select_integer(4, 'Resolution ratio (2 or 4)') 
    if not ratio:
        return        
#  MS registration band    
    k1 = auxil.select_integer(1, 'MS band for registration') 
    if not k1:
        return       
    print '========================='
    print '   ATWT Pansharpening'
    print '========================='
    print time.asctime()     
    print 'MS  file: '+file1
    print 'PAN file: '+file2       
#  image arrays
    band = inDataset1.GetRasterBand(1)
    tmp = band.ReadAsArray(0,0,1,1)
    dt = tmp.dtype
    MS = np.asarray(np.zeros((num_bands,rows1,cols1)),dtype = dt)
#  result will be float32    
    sharpened = np.zeros((num_bands,rows2,cols2),dtype=np.float32) 
    k = 0                                   
    for b in pos1:
        band = inDataset1.GetRasterBand(b)
        MS[k,:,:] = band.ReadAsArray(x10,y10,cols1,rows1)
        k += 1
    band = inDataset2.GetRasterBand(1)
    PAN = band.ReadAsArray(x20,y20,cols2,rows2) 
#  if integer assume 11bit quantization, otherwise must be byte    
    if PAN.dtype == np.int16:
        PAN = auxil.byteStretch(PAN,(0,2**11))
    if MS.dtype == np.int16:
        MS = auxil.byteStretch(MS,(0,2**11))                
#  compress PAN to resolution of MS image using DWT  
    panDWT = auxil.DWTArray(PAN,cols2,rows2)          
    r = ratio
    while r > 1:
        panDWT.filter()
        r /= 2
    bn0 = panDWT.get_quadrant(0)   
#  register (and subset) MS image to compressed PAN image using MSband  
    lines0,samples0 = bn0.shape    
    bn1 = MS[k1,:,:]  
#  register (and subset) MS image to compressed PAN image 
    (scale,angle,shift) = auxil.similarity(bn0,bn1)
    tmp = np.zeros((num_bands,lines0,samples0))
    for k in range(num_bands): 
        bn1 = MS[k,:,:]                    
        bn2 = ndii.zoom(bn1, 1.0/scale)
        bn2 = ndii.rotate(bn2, angle)
        bn2 = ndii.shift(bn2, shift)
        tmp[k,:,:] = bn2[0:lines0,0:samples0]        
    MS = tmp          
    smpl = np.random.randint(cols2*rows2,size=100000)
    print 'Wavelet correlations:'    
#  loop over MS bands
    for k in range(num_bands):
        msATWT = auxil.ATWTArray(PAN)
        r = ratio
        while r > 1:
            msATWT.filter()
            r /= 2 
#      sample PAN wavelet details
        X = msATWT.get_band(msATWT.num_iter)
        X = X.ravel()[smpl]
#      resize the ms band to scale of the pan image
        ms_band = ndii.zoom(MS[k,:,:],ratio)
#      sample details of MS band
        tmpATWT = auxil.ATWTArray(ms_band)
        r = ratio
        while r > 1:
            tmpATWT.filter()
            r /= 2                 
        Y = tmpATWT.get_band(msATWT.num_iter)
        Y = Y.ravel()[smpl]  
#      get band for injection
        bnd = tmpATWT.get_band(0) 
        tmpATWT = None 
        aa,bb,R = auxil.orthoregress(X,Y)
        print 'Band '+str(k+1)+': %8.3f'%R
#      inject the filtered MS band
        msATWT.inject(bnd)    
#      normalize wavelet components and expand
        msATWT.normalize(aa,bb)                    
        r = ratio
        while r > 1:
            msATWT.invert()
            r /= 2 
        sharpened[k,:,:] = msATWT.get_band(0)                                  
#  write to disk       
    if outfile:
        driver = gdal.GetDriverByName(fmt)   
        outDataset = driver.Create(outfile,
                        cols2,rows2,num_bands,GDT_Float32)
        projection1 = inDataset1.GetProjection()
        geotransform1 = inDataset1.GetGeoTransform()
        geotransform2 = inDataset2.GetGeoTransform()
        if geotransform2 is not None:
            gt2 = list(geotransform2)
            if geotransform1 is not None:
                gt1 = list(geotransform1)
                gt1[0] += x10*gt2[1]  # using PAN pixel sizes
                gt1[3] += y10*gt2[5]
                gt1[1] = gt2[1]
                gt1[2] = gt2[2]
                gt1[4] = gt2[4]
                gt1[5] = gt2[5]
                outDataset.SetGeoTransform(tuple(gt1))
        if projection1 is not None:
            outDataset.SetProjection(projection1)        
        for k in range(num_bands):        
            outBand = outDataset.GetRasterBand(k+1)
            outBand.WriteArray(sharpened[k,:,:],0,0) 
            outBand.FlushCache() 
        outDataset = None    
    print 'Result written to %s'%outfile    
    inDataset1 = None
    inDataset2 = None                      

if __name__ == '__main__':
    main()