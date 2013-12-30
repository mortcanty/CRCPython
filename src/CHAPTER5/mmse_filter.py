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
import auxil.polsar as polsar
import auxil.congrid as congrid
import os, time
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GDT_CFloat32

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

#def get_windex(i,cols):
##  first window for column i    
#    windex = np.zeros(49,dtype=int)
#    six = np.array([0,1,2,3,4,5,6])*cols
#    windex[0:7]   = (i-3) + six
#    windex[7:14]  = (i-2) + six
#    windex[14:21] = (i-1) + six
#    windex[21:28] = (i)   + six 
#    windex[28:35] = (i+1) + six 
#    windex[35:42] = (i+2) + six
#    windex[42:49] = (i+3) + six
#    return windex

def main():
    gdal.AllRegister()
    path = auxil.select_directory('Choose working directory')
    if path:
        os.chdir(path)        
#  SAR image    
    infile = auxil.select_infile(filt='.hdr',title='Select SAR image header') 
    if not infile:
        return
    end = auxil.select_integer(1,msg='Byte order: 1=Little Endian, 2=Big Endian')
    if end == 2:
        endian = 'B'
    else:
        endian = 'L'
    ps = polsar.Polsar(infile,endian=endian)
#  number of looks
    m = auxil.select_integer(5,msg='Number of looks')
    if not m:
        return
#  output file
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return       
#  get filter weights from span image
    rows = ps.lines
    cols = ps.samples
    b = np.ones((rows,cols))
    span = ps.span().ravel()
    edge_idx = np.zeros((rows,cols),dtype=int)
    print '========================='
    print '       MMSE_FILTER'
    print '========================='
    print 'infile:  %s'%infile
    print 'number of looks: %i'%m     
    print 'Determining filter weights from span image'    
    start = time.time()     
    for j in range(3,rows-3):
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
            b[j,i] = np.max( ((1.0 - gbar**2/(varg*m))/(1.0+1.0/m), 0.0) )        
            windex += 1
#  filter the image
    outim = np.zeros((rows,cols),dtype=complex)
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,6,GDT_CFloat32)
    if ps.geotransform is not None:
        outDataset.SetGeoTransform(ps.geotransform)
    if ps.projection is not None:
        outDataset.SetProjection(ps.projection) 
    print 'Filtering covariance matrix elememnts'  
    for k in range(6):
        print 'band: %i'%(k+1)
        band = ps.band(k)
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
        outim = gbar + b*(band-gbar)   
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(outim,0,0) 
        outBand.FlushCache() 
    outDataset = None
    print 'result written to: '+outfile 
    print 'elapsed time: '+str(time.time()-start)                 
              
if __name__ == '__main__':
    main()
    