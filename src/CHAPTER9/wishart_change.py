#!/usr/bin/env python
#******************************************************************************
#  Name:     wishart_change.py
#  Purpose:  Perfrom change detection on bitemporal, polarimetric EMISAR imagery 
#            Based on Allan Nielsen's Matlab script
#  Usage:             
#    python wishart_change.py 
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
import registerSAR
import numpy as np
from scipy import stats
import os, time, gdal 
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32

                       
def main():
    print '================================'
    print 'Complex Wishart Change Detection'
    print '================================'
    print time.asctime()
    gdal.AllRegister()
    path = auxil.select_directory('Choose working directory')
    if path:
        os.chdir(path)        
#  first SAR image    
    infile1 = auxil.select_infile(title='Choose first SAR image') 
    if infile1:                   
        inDataset1 = gdal.Open(infile1,GA_ReadOnly)     
        cols = inDataset1.RasterXSize
        rows = inDataset1.RasterYSize    
        bands = inDataset1.RasterCount
    else:
        return
    m = auxil.select_integer(5,msg='Number of looks')
    if not m:
        return
    print 'first filename:  %s'%infile1
    print 'number of looks: %i'%m  
#  second SAR image    
    infile2 = auxil.select_infile(title='Choose second SAR image') 
    if not infile2:                   
        return
    n = auxil.select_integer(5,msg='Number of looks')
    if not n:
        return
    print 'second filename:  %s'%infile2
    print 'number of looks: %i'%n  
#  output file
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return    
    print 'co-registering...'
    registerSAR.registerSAR(infile1,infile2,'warp.tif','GTiff')
    infile2 = 'warp.tif'
    inDataset2 = gdal.Open(infile2,GA_ReadOnly)     
    cols2 = inDataset2.RasterXSize
    rows2 = inDataset2.RasterYSize    
    bands2 = inDataset2.RasterCount   
    if (bands != bands2) or (cols != cols2) or (rows != rows2):
        print 'Size mismatch'
        return    
    start = time.time() 
    if bands == 9:
        print 'Quad polarimetry'  
#      C11 (k1)
        b = inDataset1.GetRasterBand(1)
        k1 = m*b.ReadAsArray(0,0,cols,rows)
#      C12  (a1)
        b = inDataset1.GetRasterBand(2)
        a1 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset1.GetRasterBand(3)    
        im = b.ReadAsArray(0,0,cols,rows)
        a1 = m*(a1 + 1j*im)
#      C13  (rho1)
        b = inDataset1.GetRasterBand(4)
        rho1 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset1.GetRasterBand(5)
        im = b.ReadAsArray(0,0,cols,rows)
        rho1 = m*(rho1 + 1j*im)      
#      C22 (xsi1)
        b = inDataset1.GetRasterBand(6)
        xsi1 = m*b.ReadAsArray(0,0,cols,rows)    
#      C23 (b1)        
        b = inDataset1.GetRasterBand(7)
        b1 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset1.GetRasterBand(8)
        im = b.ReadAsArray(0,0,cols,rows)
        b1 = m*(b1 + 1j*im)      
#      C33 (zeta1)
        b = inDataset1.GetRasterBand(9)
        zeta1 = m*b.ReadAsArray(0,0,cols,rows)              
#      C11 (k2)
        b = inDataset2.GetRasterBand(1)
        k2 = n*b.ReadAsArray(0,0,cols,rows)
#      C12  (a2)
        b = inDataset2.GetRasterBand(2)
        a2 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset2.GetRasterBand(3)
        im = b.ReadAsArray(0,0,cols,rows)
        a2 = n*(a2 + 1j*im)
#      C13  (rho2)
        b = inDataset2.GetRasterBand(4)
        rho2 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset2.GetRasterBand(5)
        im = b.ReadAsArray(0,0,cols,rows)
        rho2 = n*(rho2 + 1j*im)        
#      C22 (xsi2)
        b = inDataset2.GetRasterBand(6)
        xsi2 = n*b.ReadAsArray(0,0,cols,rows)    
#      C23 (b2)        
        b = inDataset2.GetRasterBand(7)
        b2 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset2.GetRasterBand(8)
        im = b.ReadAsArray(0,0,cols,rows)
        b2 = n*(b2 + 1j*im)        
#      C33 (zeta2)
        b = inDataset2.GetRasterBand(9)
        zeta2 = n*b.ReadAsArray(0,0,cols,rows)           
        k3    = k1 + k2  
        a3    = a1 + a2
        rho3  = rho1 + rho2
        xsi3  = xsi1 + xsi2
        b3    = b1 + b2
        zeta3 = zeta1 + zeta2           
        det1 = k1*xsi1*zeta1 + 2*np.real(a1*b1*np.conj(rho1)) - xsi1*(abs(rho1)**2) - k1*(abs(b1)**2) - zeta1*(abs(a1)**2)    
        det2 = k2*xsi2*zeta2 + 2*np.real(a2*b2*np.conj(rho2)) - xsi2*(abs(rho2)**2) - k2*(abs(b2)**2) - zeta2*(abs(a2)**2)       
        det3 = k3*xsi3*zeta3 + 2*np.real(a3*b3*np.conj(rho3)) - xsi3*(abs(rho3)**2) - k3*(abs(b3)**2) - zeta3*(abs(a3)**2)       
        p = 3
        f = p**2
        cst = p*((n+m)*np.log(n+m)-n*np.log(n)-m*np.log(m)) 
        rho = 1. - (2.*p**2-1.)*(1./n + 1./m - 1./(n+m))/(6.*p)    
        omega2 = -(p*p/4.)*(1. - 1./rho)**2 + p**2*(p**2-1.)*(1./n**2 + 1./m**2 - 1./(n+m)**2)/(24.*rho**2)        
    elif bands == 4:
        print 'Dual polarimetry'  
#      C11 (k1)
        b = inDataset1.GetRasterBand(1)
        k1 = m*b.ReadAsArray(0,0,cols,rows)
#      C12  (a1)
        b = inDataset1.GetRasterBand(2)
        a1 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset1.GetRasterBand(3)
        im = b.ReadAsArray(0,0,cols,rows)
        a1 = m*(a1 + 1j*im)        
#      C22 (xsi1)
        b = inDataset1.GetRasterBand(4)
        xsi1 = m*b.ReadAsArray(0,0,cols,rows)          
#      C11 (k2)
        b = inDataset2.GetRasterBand(1)
        k2 = n*b.ReadAsArray(0,0,cols,rows)
#      C12  (a2)
        b = inDataset2.GetRasterBand(2)
        a2 = b.ReadAsArray(0,0,cols,rows)
        b = inDataset2.GetRasterBand(3)
        im = b.ReadAsArray(0,0,cols,rows)
        a2 = n*(a2 + 1j*im)        
#      C22 (xsi2)
        b = inDataset2.GetRasterBand(4)
        xsi2 = n*b.ReadAsArray(0,0,cols,rows)        
        k3    = k1 + k2  
        a3    = a1 + a2
        xsi3  = xsi1 + xsi2       
        det1 = k1*xsi1 - abs(a1)**2
        det2 = k2*xsi2 - abs(a2)**2 
        det3 = k3*xsi3 - abs(a3)**2        
        p = 2 
        cst = p*((n+m)*np.log(n+m)-n*np.log(n)-m*np.log(m)) 
        f = p**2
        rho = 1-(2*f-1)*(1./n+1./m-1./(n+m))/(6.*p)
        omega2 = -f/4.*(1-1./rho)**2 + f*(f-1)*(1./n**2+1./m**2-1./(n+m)**2)/(24.*rho**2)  
    elif bands == 1:
        print 'Single polarimetry'         
#      C11 (k1)
        b = inDataset1.GetRasterBand(1)
        k1 = m*b.ReadAsArray(0,0,cols,rows) 
#      C11 (k2)
        b = inDataset2.GetRasterBand(1)
        k2 = n*b.ReadAsArray(0,0,cols,rows) 
        k3 = k1 + k2
        det1 = k1 
        det2 = k2
        det3 = k3    
        p = 1 
        cst = p*((n+m)*np.log(n+m)-n*np.log(n)-m*np.log(m)) 
        f = p**2
        rho = 1-(2.*f-1)*(1./n+1./m-1./(n+m))/(6.*p)
        omega2 = -f/4.*(1-1./rho)**2+f*(f-1)*(1./n**2+1./m**2-1./(n+m)**2)/(24.*rho**2)  
    else:   
        print 'Incorrect number of bands'
        return   
    idx = np.where(det1 <= 0.0)
    det1[idx] = 0.0001   
    idx = np.where(det2 <= 0.0)
    det2[idx] = 0.0001 
    idx = np.where(det3 <= 0.0)
    det3[idx] = 0.0001  
    lnQ = cst+m*np.log(det1)+n*np.log(det2)-(n+m)*np.log(det3)
#  test statistic    
    Z = -2*rho*lnQ
#  change probabilty
    P =  (1.-omega2)*stats.chi2.cdf(Z,[f])+omega2*stats.chi2.cdf(Z,[f+4])
#  write to file system        
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,2,GDT_Float32)
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    projection = inDataset1.GetProjection()        
    if projection is not None:
        outDataset.SetProjection(projection) 
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(Z,0,0) 
    outBand.FlushCache() 
    outBand = outDataset.GetRasterBand(2)
    outBand.WriteArray(P,0,0) 
    outBand.FlushCache()     
    outDataset = None
    print 'result written to: %s'%outfile 
    print 'elapsed time: '+str(time.time()-start)      
                
if __name__ == '__main__':
    main()     