#!/usr/bin/env python
#******************************************************************************
#  Name:     polsar.py
#  Purpose: Object class to store fully polarized SAR data in multilook covariance matrix form.
#     Property PLANES (sum of real and complex planes) identifies polarization type:
#        9: full polarimetry (upper diagonal part of 3x3 matrix: 3 real bands, 3 complex bands)
#        5: full polarimetry, azimuthal symmetry
#        3: full polarimetry, diagonal only
#        4: dual polarimetry
#        2: dual polarimetry, diagonal only
#        1: one-channel data  
#  Usage:             
#    import polsar
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

from numpy import * 
from osgeo.gdalconst import GA_ReadOnly
import os, header, gdal                      

class Polsar(object):  
       
    def __init__(self,fname,endian='L'):
        path = os.path.dirname(fname)
        basename = os.path.basename(fname)
        root, _ = os.path.splitext(basename)   
        f = open(fname)
        line = f.readline()
        f.close()
        if line != 'ENVI\n':
#          not an ENVI standard file, so assume DLR format
#                        (mixed real and complex bands)
            hdr = header.Header()
            f = open(fname)
            hdr.read(f.read())
            samples = int(hdr['samples'])
            lines = int(hdr['lines'])
            self.bands = int(hdr['bands'])
            self.lines = lines
            self.samples = samples
            f.close()     
            planes = 0
            dtr = dtype(float32).newbyteorder(endian)
            dtc = dtype(complex64).newbyteorder(endian)
            fn = path+'\\'+root+'hhhh'
            if os.path.exists(fn):
                a = reshape(fromfile(fn,dtr),(lines,samples)) 
                planes += 1
            else:
                a = zeros((lines,samples),dtype=dtr)  
            self.c11 = a     
            fn = path+'\\'+root+'hhhv'             
            if os.path.exists(fn):
                a = reshape(fromfile(fn,dtc),(lines,samples)) 
                planes += 2
            else:
                a = zeros((lines,samples),dtype=dtc)  
            self.c12 = a            
            fn = path+'\\'+root+'hhvv'             
            if os.path.exists(fn):
                a = reshape(fromfile(fn,dtc),(lines,samples)) 
                planes += 2
            else:
                a = zeros((lines,samples),dtype=dtc) 
            self.c13 = a 
            fn = path+'\\'+root+'hvhv'             
            if os.path.exists(fn):
                a = reshape(fromfile(fn,dtr),(lines,samples)) 
                planes += 1
            else:
                a = zeros((lines,samples),dtype=dtr)  
            self.c22 = a     
            fn = path+'\\'+root+'hvvv'             
            if os.path.exists(fn):
                a = reshape(fromfile(fn,dtc),(lines,samples)) 
                planes += 2
            else:
                a = zeros((lines,samples),dtype=dtc) 
            self.c23 = a     
            fn = path+'\\'+root+'vvvv'             
            if os.path.exists(fn):
                a = reshape(fromfile(fn,dtr),(lines,samples)) 
                planes += 1
            else:
                a = zeros((lines,samples),dtype=dtr)  
            self.c33 = a 
            self.geotransform = None
            self.projection = None
        else:
#          ENVI standard file, 6 bands, complex data
            basename, ext = os.path.splitext(fname) 
            inDataset = gdal.Open(basename,GA_ReadOnly)
            self.samples = inDataset.RasterXSize
            self.lines = inDataset.RasterYSize    
            self.bands = inDataset.RasterCount
            self.geotransform = inDataset.GetGeoTransform()
            self.projection = inDataset.GetProjection()
            if self.bands != 6:
                return
            planes = 0
            band = inDataset.GetRasterBand(1)
            self.c11 = real(band.ReadAsArray(0,0,self.samples,self.lines))
            if sum(self.c11) != 0:
                planes += 1
            band = inDataset.GetRasterBand(2)
            self.c12 = band.ReadAsArray(0,0,self.samples,self.lines)\
                              .astype(complex)
            if sum(self.c12) != 0:
                planes += 2
            band = inDataset.GetRasterBand(3)
            self.c13 = band.ReadAsArray(0,0,self.samples,self.lines)\
                              .astype(complex)
            if sum(self.c13) != 0:
                planes += 2                
            band = inDataset.GetRasterBand(4)
            self.c22 = real(band.ReadAsArray(0,0,self.samples,self.lines))
            if sum(self.c22) != 0:
                planes += 1          
            band = inDataset.GetRasterBand(5)
            self.c23 = band.ReadAsArray(0,0,self.samples,self.lines)\
                              .astype(complex)
            if sum(self.c23) != 0:
                planes += 2                        
            band = inDataset.GetRasterBand(6)
            self.c33 = real(band.ReadAsArray(0,0,self.samples,self.lines))
            if sum(self.c33) != 0:
                planes += 1  
        self.planes = planes                                                             
        self.path = path
        self.root = root
        return None
    
    def span(self):
        return self.c11 + 2*self.c22 + self.c33
    
    def band(self,k):
        if k==0:
            return self.c11
        elif k==1:
            return self.c12
        elif k==2:
            return self.c13
        elif k==3:
            return self.c22
        elif k==4:
            return self.c23
        else:
            return self.c33
        
        
    def times(self,n):
        self.c11 *= n
        self.c12 *= n
        self.c13 *= n
        self.c22 *= n
        self.c23 *= n
        self.c33 *= n