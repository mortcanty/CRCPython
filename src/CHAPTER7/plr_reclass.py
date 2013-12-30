#!/usr/bin/env python
#******************************************************************************
#  Name:     plr_reclass.py
#  Purpose:  Probabilistic label relaxation reclassification 
#  Usage:             
#    python plr_reclass.py 
#
#  Copyright (c) 2012, Mort Canty
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
import auxil.header as header
import os, time
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
    
def main():
    gdal.AllRegister()
    path = auxil.select_directory('Choose working directory')
#    path = 'd:\\imagery\\CRC\\Chapters6-7'
    if path:
        os.chdir(path)   
    infile = auxil.select_infile(title='Select a class probability image') 
    if infile:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        K = inDataset.RasterCount
    else:
        return
    outfile, fmt = auxil.select_outfilefmt()  
    if not outfile:
        return   
    print '========================='
    print '       PLR_reclass'
    print '========================='
    print 'infile:  %s'%infile
    start = time.time() 
    prob_image = np.zeros((K,rows,cols))
    for k in range (K):
        band = inDataset.GetRasterBand(k+1)
        prob_image[k,:,:] = band.ReadAsArray(0,0,cols,rows).astype(float)                                   
    class_image = np.zeros((rows,cols),dtype=np.byte)  
    print 'reclassifying...'
    for i in range(rows):
        if i % 50 == 0:
            print '%i rows processed'%i
        for j in range(cols):
            cls = np.where(prob_image[:,i,j]==np.amax(prob_image[:,i,j]))[0][0]
            if isinstance(cls,int):
                class_image[i,j] = cls+1               
#  write to disk
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Byte)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection)               
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(class_image,0,0) 
    outBand.FlushCache() 
    outDataset = None
    inDataset = None
    if (fmt == 'ENVI') and (K<19):
#          try to make an ENVI classification header file 
        classnames = '{unclassified '   
        for i in range(K):
            classnames += ', '+str(i+1)
        classnames += '}'       
        hdr = header.Header() 
        headerfile = outfile+'.hdr'
        f = open(headerfile)
        line = f.readline()
        envihdr = ''
        while line:
            envihdr += line
            line = f.readline()
        f.close()         
        hdr.read(envihdr)
        hdr['file type'] ='ENVI Classification'
        hdr['classes'] = str(K+1)
        classlookup = '{0'
        for i in range(1,3*(K+1)):
            classlookup += ', '+str(str(auxil.ctable[i]))
        classlookup +='}'    
        hdr['class lookup'] = classlookup
        hdr['class names'] = classnames
        f = open(headerfile,'w')
        f.write(str(hdr))
        f.close()       
    print 'result written to: '+outfile    
    print 'elapsed time: '+str(time.time()-start)                        
    print '--done------------------------'  
       
if __name__ == '__main__':
    main()    