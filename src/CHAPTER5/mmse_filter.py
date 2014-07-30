#!/usr/bin/env python
#******************************************************************************
#  Name:     mmse_filter.py
#  Purpose: Lee MMSE adaptive filtering 
#    for polSAR covariance images
#    Lee et al. (1999) IEEE TGARS 37(5), 2363-2373
#    Oliver and Quegan (2004) Understanding SAR Images, Scitech 
#  Usage:             
#    python mmse_filter.py 
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
import auxil.congrid as congrid
import os, sys, time
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32

templates = np.zeros((8,7,7),dtype=int)
for j in range(7):
    templates[0,j,0:3] = 1
templates[1,1,0]  = 1
templates[1,2,:2] = 1
templates[1,3,:3] = 1
templates[1,4,:4] = 1
templates[1,5,:5] = 1
templates[1,6,:6] = 1
templates[2] = np.rot90(templates[0])
templates[3] = np.rot90(templates[1])
templates[4] = np.rot90(templates[2])
templates[5] = np.rot90(templates[3])
templates[6] = np.rot90(templates[4])
templates[7] = np.rot90(templates[5])

tmp = np.zeros((8,21),dtype=int)
for i in range(8):
    tmp[i,:] = np.where(templates[i].ravel())[0] 
templates = tmp

edges = np.zeros((4,3,3),dtype=int)
edges[0] = [[-1,0,1],[-1,0,1],[-1,0,1]]
edges[1] = [[0,1,1],[-1,0,1],[-1,-1,0]]
edges[2] = [[1,1,1],[0,0,0],[-1,-1,-1]]
edges[3] = [[1,1,0],[1,0,-1],[0,-1,-1]]   
    
   
def get_windex(j,cols):
#  first window for row j    
    windex = np.zeros(49,dtype=int)
    six = np.array([0,1,2,3,4,5,6])
    windex[0:7]   = (j-3)*cols + six
    windex[7:14]  = (j-2)*cols + six
    windex[14:21] = (j-1)*cols + six
    windex[21:28] = (j)*cols   + six 
    windex[28:35] = (j+1)*cols + six 
    windex[35:42] = (j+2)*cols + six
    windex[42:49] = (j+3)*cols + six
    return windex

def main():
    gdal.AllRegister()
    path = auxil.select_directory('Choose working directory')
    if path:
        os.chdir(path)        
#  SAR image    
    infile = auxil.select_infile(title='Choose SAR image') 
    if infile:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    else:
        return
#  spatial subset    
    x0,y0,rows,cols=auxil.select_dims([0,0,rows,cols])    
#  number of looks
    m = auxil.select_integer(5,msg='Number of looks')
    if not m:
        return
#  output file
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return       
#  get filter weights from span image
    b = np.ones((rows,cols))
    band = inDataset.GetRasterBand(1)
    span = band.ReadAsArray(x0,y0,cols,rows).ravel()
    if bands==9:      
        band = inDataset.GetRasterBand(6)
        span += band.ReadAsArray(x0,y0,cols,rows).ravel()
        band = inDataset.GetRasterBand(9)
        span += band.ReadAsArray(x0,y0,cols,rows).ravel()
    elif bands==4:
        band = inDataset.GetRasterBand(4)
        span += band.ReadAsArray(x0,y0,cols,rows).ravel()    
    edge_idx = np.zeros((rows,cols),dtype=int)
    print '========================='
    print '       MMSE_FILTER'
    print '========================='
    print time.asctime()
    print 'infile:  %s'%infile
    print 'number of looks: %i'%m     
    print 'Determining filter weights from span image'    
    start = time.time()
    print 'row: ',
    sys.stdout.flush()     
    for j in range(3,rows-3):
        if j%50 == 0:
            print '%i '%j, 
            sys.stdout.flush()
        windex = get_windex(j,cols)
        for i in range(3,cols-3):            
            wind = np.reshape(span[windex],(7,7))         
#          3x3 compression
            w = congrid.congrid(wind,(3,3),method='spline',centre=True)
#          get appropriate edge mask
            es = [np.sum(edges[p]*w) for p in range(4)]
            idx = np.argmax(es)  
            if idx == 0:
                if np.abs(w[1,1]-w[1,0]) < np.abs(w[1,1]-w[1,2]):
                    edge_idx[j,i] = 0
                else:
                    edge_idx[j,i] = 4
            elif idx == 1:
                if np.abs(w[1,1]-w[2,0]) < np.abs(w[1,1]-w[0,2]):
                    edge_idx[j,i] = 1
                else:
                    edge_idx[j,i] = 5                
            elif idx == 2:
                if np.abs(w[1,1]-w[0,1]) < np.abs(w[1,1]-w[2,1]):
                    edge_idx[j,i] = 6
                else:
                    edge_idx[j,i] = 2  
            elif idx == 3:
                if np.abs(w[1,1]-w[0,0]) < np.abs(w[1,1]-w[2,2]):
                    edge_idx[j,i] = 7
                else:
                    edge_idx[j,i] = 3 
            edge = templates[edge_idx[j,i]]  
            wind = wind.ravel()[edge]
            gbar = np.mean(wind)
            varg = np.var(wind)
            if varg > 0:
                b[j,i] = np.max( ((1.0 - gbar**2/(varg*m))/(1.0+1.0/m), 0.0) )        
            windex += 1
    print ' done'        
#  filter the image
    outim = np.zeros((rows,cols),dtype=np.float32)
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,bands,GDT_Float32)
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    projection = inDataset.GetProjection()        
    if projection is not None:
        outDataset.SetProjection(projection) 
    print 'Filtering covariance matrix elememnts'  
    for k in range(1,bands+1):
        print 'band: %i'%(k)
        band = inDataset.GetRasterBand(k)
        band = band.ReadAsArray(0,0,cols,rows)
        gbar = band*0.0
#      get window means
        for j in range(3,rows-3):        
            windex = get_windex(j,cols)
            for i in range(3,cols-3):
                wind = band.ravel()[windex]
                edge = templates[edge_idx[j,i]]
                wind = wind[edge]
                gbar[j,i] = np.mean(wind)
                windex += 1
#      apply adaptive filter and write to disk
        outim = np.reshape(gbar + b*(band-gbar),(rows,cols))   
        outBand = outDataset.GetRasterBand(k)
        outBand.WriteArray(outim,0,0) 
        outBand.FlushCache() 
    outDataset = None
    print 'result written to: '+outfile 
    print 'elapsed time: '+str(time.time()-start)                 
              
if __name__ == '__main__':
    main()
    