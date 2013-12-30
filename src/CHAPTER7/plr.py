#!/usr/bin/env python
#******************************************************************************
#  Name:     plr.py
#  Purpose:  Probabilistic label relaxation 
#  Usage:             
#    python plr.py 
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
        classes = inDataset.RasterCount
    else:
        return
    outfile, fmt = auxil.select_outfilefmt()  
    if not outfile:
        return   
    nitr = auxil.select_integer(3,'Select number of iterations')
    print '========================='
    print '       PLR'
    print '========================='
    print 'infile:  %s'%infile
    print 'iterations:  %i'%nitr
    start = time.time()                                     
    prob_image = np.zeros((classes,rows,cols))
    for k in range (classes):
        band = inDataset.GetRasterBand(k+1)
        prob_image[k,:,:] = band.ReadAsArray(0,0,cols,rows).astype(float)
#  compatibility matrix
    Pmn = np.zeros((classes,classes))
    n_samples = (cols-1)*(rows-1)
    samplem = np.reshape(prob_image[:,0:rows-1,0:cols-1],(classes,n_samples))
    samplen = np.reshape(prob_image[:,1:rows,0:cols-1],(classes,n_samples))
    sampleu = np.reshape(prob_image[:,0:rows-1,1:cols],(classes,n_samples))
    max_samplem = np.amax(samplem,axis=0)
    max_samplen = np.amax(samplen,axis=0)
    max_sampleu = np.amax(sampleu,axis=0)
    print 'estimating compatibility matrix...'
    for j in range(n_samples):
        if j % 50000 == 0:
            print '%i samples of %i'%(j,n_samples)
        m1 = np.where(samplem[:,j]==max_samplem[j])[0][0]
        n1 = np.where(samplen[:,j]==max_samplen[j])[0][0]
        if isinstance(m1,int) and isinstance(n1,int):
            Pmn[m1,n1] += 1
        u1 = np.where(sampleu[:,j]==max_sampleu[j])[0][0] 
        if isinstance(m1,int) and isinstance(u1,int): 
            Pmn[m1,u1] += 1  
    for j in range(classes):
        n = np.sum(Pmn[j,:])
        if n>0:
            Pmn[j,:] /= n        
    print Pmn    
    itr = 0
    temp = prob_image*0
    print 'label relaxation...'
    while itr<nitr:
        print 'iteration %i'%(itr+1)
        Pm = np.zeros(classes)
        Pn = np.zeros(classes)
        for i in range(1,rows-1):
            if i % 50 == 0:
                print '%i rows processed'%i
            for j in range(1,cols-1):
                Pm[:] = prob_image[:,i,j]
                Pn[:] = prob_image[:,i-1,j]/4
                Pn[:] += prob_image[:,i+1,j]/4
                Pn[:] += prob_image[:,i,j-1]/4
                Pn[:] += prob_image[:,i,j+1]/4
                Pn = np.transpose(Pn)
                if np.sum(Pm) == 0:
                    Pm_new = Pm
                else:
                    Pm_new = Pm*(np.dot(Pmn,Pn))/(np.dot(np.dot(Pm,Pmn),Pn))
                temp[:,i,j] = Pm_new
        prob_image = temp
        itr += 1    
#  write to disk
    prob_image = np.byte(prob_image*255)
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,classes,GDT_Byte)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection)        
    for k in range(classes):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(prob_image[k,:,:],0,0) 
        outBand.FlushCache() 
    outDataset = None
    inDataset = None
    print 'result written to: '+outfile    
    print 'elapsed time: '+str(time.time()-start)                        
    print '--done------------------------'  
       
if __name__ == '__main__':
    main()    