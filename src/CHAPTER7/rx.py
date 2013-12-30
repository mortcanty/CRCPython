#!/usr/bin/env python
#******************************************************************************
#  Name:     rx.py
#  Purpose:  RX anomaly detection for multi- and hyperspectral images
#  Usage:             
#    python rx.py
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
import gdal, os
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
import numpy as np 
from spectral.algorithms.detectors import RX 
from spectral.algorithms.algorithms import calc_stats 
import spectral.io.envi as envi
 
def main():      
    gdal.AllRegister()
    path = auxil.select_directory('Input directory')
    if path:
        os.chdir(path)        
#  input image, convert to ENVI format   
    infile = auxil.select_infile(title='Image file') 
    if infile:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize          
        projection = inDataset.GetProjection()
        geotransform = inDataset.GetGeoTransform()       
        driver = gdal.GetDriverByName('ENVI')        
        enviDataset=driver\
           .CreateCopy('entmp',inDataset)
        inDataset = None
        enviDataset = None  
    else:
        return  
    outfile, outfmt= \
           auxil.select_outfilefmt(title='Output file')   
#  RX-algorithm        
    img = envi.open('entmp.hdr')
    arr = img.load()
    rx = RX(background=calc_stats(arr))
    res = rx(arr)
#  output 
    driver = gdal.GetDriverByName(outfmt)    
    outDataset = driver.Create(outfile,cols,rows,1,\
                                    GDT_Float32) 
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection) 
    outBand = outDataset.GetRasterBand(1) 
    outBand.WriteArray(np.asarray(res,np.float32),0,0) 
    outBand.FlushCache()
    outDataset = None   
    print 'Result written to %s'%outfile
    
if __name__ == '__main__':
    main()