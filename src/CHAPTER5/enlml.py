#!/usr/bin/env python
#******************************************************************************
#  Name:     enlml.py
#  Purpose: 
#    Estimation of ENL for polSAR covariance images
#    using ML method with full covariance matrix (quad, dual or single)
#    Anfinsen et al. (2009) IEEE TGARS 47(11), 3795-3809
#    Takes input from covariance matrix format images generated
#    from polsaringest.py
#  Usage:             
#    python enl.py 
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
import auxil.lookup as lookup
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
   
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
#  output file
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return       
    print '========================='
    print '     ENL Estimation'
    print '========================='
    print time.asctime()
    print 'infile:  %s'%infile   
    start = time.time()
    if bands == 9:
        print 'Quad polarimetry'  
#      C11 (k)
        band = inDataset.GetRasterBand(1)
        k = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
#      C12  (a)
        band = inDataset.GetRasterBand(2)
        a = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        band = inDataset.GetRasterBand(3)    
        im = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        a = a + 1j*im
#      C13  (rho)
        band = inDataset.GetRasterBand(4)
        rho = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        band = inDataset.GetRasterBand(5)
        im = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        rho = rho + 1j*im     
#      C22 (xsi)
        band = inDataset.GetRasterBand(6)
        xsi = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()    
#      C23 (b)        
        band = inDataset.GetRasterBand(7)
        b = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        band = inDataset.GetRasterBand(8)
        im = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        b = b + 1j*im     
#      C33 (zeta)
        band = inDataset.GetRasterBand(9)
        zeta = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()                
        det = k*xsi*zeta + 2*np.real(a*b*np.conj(rho)) - xsi*(abs(rho)**2) - k*(abs(b)**2) - zeta*(abs(a)**2)
        d = 2
    elif bands == 4:
        print 'Dual polarimetry'  
#      C11 (k)
        band = inDataset.GetRasterBand(1)
        k = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
#      C12  (a)
        band = inDataset.GetRasterBand(2)
        a = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        band = inDataset.GetRasterBand(3)
        im = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel()
        a = a + 1j*im       
#      C22 (xsi)
        band = inDataset.GetRasterBand(4)
        xsi = np.nan_to_num(band.ReadAsArray(x0,y0,cols,rows)).ravel() 
        det = k*xsi - abs(a)**2   
        d = 1   
    elif bands == 1:
        print 'Single polarimetry'         
#      C11 (k)
        band = inDataset.GetRasterBand(1)
        k = band.ReadAsArray(x0,y0,cols,rows).ravel() 
        det = k
        d = 0      
    enl_ml = np.zeros((rows,cols), dtype= np.float32)
    lu = lookup.table()
    print 'filtering...'
    print 'row: ',
    sys.stdout.flush()    
    start = time.time()
    for i in range(3,rows-3):
        if i%50 == 0:
            print '%i '%i, 
            sys.stdout.flush()
        windex = get_windex(i,cols)  
        for j in range(3,cols-3):  
            detC = det[windex]
            if np.min(detC) > 0.0:
                avlogdetC = np.sum(np.log(detC))/49
                if bands == 9:
                    k1 = np.sum(k[windex])/49
                    a1 = np.sum(a[windex])/49
                    rho1 = np.sum(rho[windex])/49
                    xsi1 = np.sum(xsi[windex])/49
                    b1 = np.sum(b[windex])/49
                    zeta1 = np.sum(zeta[windex])/49
                    detavC = k1*xsi1*zeta1 + 2*np.real(a1*b1*np.conj(rho1)) - xsi1*(np.abs(rho1)**2) - k1*(np.abs(b1)**2) - zeta1*(np.abs(a1)**2)
                elif bands == 4:
                    k1 = np.sum(k[windex])/49
                    xsi1 = np.sum(xsi[windex])/49
                    a1 = np.sum(a[windex])/49   
                    detavC =  k1*xsi1 - np.abs(a1)**2
                else:
                    detavC = np.sum(k[windex])/49
                logdetavC = np.log(detavC)    
                arr =  avlogdetC - logdetavC + lu[:,d]    
                ell = np.where(arr*np.roll(arr,1)<0)[0]
                if ell != []:
                    enl_ml[i,j] = float(ell[-1])/10.0
            windex += 1
    driver = gdal.GetDriverByName(fmt)   
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)          
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(enl_ml,0,0) 
    outBand.FlushCache() 
    outDataset = None   
    ya,xa = np.histogram(enl_ml,bins=50)
    ya[0] = 0    
    plt.plot(xa[0:-1],ya)
    plt.show() 
    print ''        
    print 'ENL image written to: %s'%outfile                  
    print 'elapsed time: '+str(time.time()-start)                    
        
if __name__ == '__main__':
    main()
    