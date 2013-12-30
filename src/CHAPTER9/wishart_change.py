#!/usr/bin/env python
#******************************************************************************
#  Name:     wishart_change.py
#  Purpose:  Perfrom change detection on bitemporal, polarimetric EMISAR imagery 
#            Based on Allan Nielsen's Matlab script
#  Usage:             
#    python wishart_change.py filenamex filenamey n m
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
import auxil.polsar as polsar
from numpy import * 
from scipy import stats
import os, time, gdal 
from osgeo.gdalconst import GDT_Float32

                       
def main():
    print '================================'
    print 'Complex Wishart Change Detection'
    print '================================'
    gdal.AllRegister()
    path = auxil.select_directory('Choose working directory')
    if path:
        os.chdir(path)        
#  first SAR image    
    infile = auxil.select_infile(filt='.hdr',title='Select first SAR image header') 
    if not infile:
        return
    end = auxil.select_integer(1,msg='Byte order: 1=Little Endian, 2=Big Endian')
    if end == 2:
        endian = 'B'
    else:
        endian = 'L'
    m = auxil.select_integer(5,msg='Number of looks')
    if not m:
        return
    print 
    print 'first filename:  %s'%infile
    print 'number of looks: %i'%m  
    X = polsar.Polsar(infile,endian=endian)
#  second SAR image    
    infile = auxil.select_infile(filt='.hdr',title='Select second SAR image header') 
    if not infile:
        return
    end = auxil.select_integer(1,msg='Byte order: 1=Little Endian, 2=Big Endian')
    if end == 2:
        endian = 'B'
    else:
        endian = 'L'
    n = auxil.select_integer(5,msg='Number of looks')
    if not n:
        return
    print 'second filename:  %s'%infile
    print 'number of looks: %i'%n  
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return      
    Y = polsar.Polsar(infile,endian=endian)  
    if (X.bands != Y.bands) or (X.lines != Y.lines) or (X.samples != Y.samples):
        print 'Size mismatch'
        return
    if  X.planes != Y.planes:
        print 'File mismatch'
        return        
    X.times(n)
    Y.times(m)            
    k1    = X.c11
    a1    = X.c12
    rho1  = X.c13
    xsi1  = X.c22
    b1    = X.c23
    zeta1 = X.c33    
    k2    = Y.c11
    a2    = Y.c12
    rho2  = Y.c13
    xsi2  = Y.c22
    b2    = Y.c23
    zeta2 = Y.c33 
    k3    = k1 + k2  
    a3    = a1 + a2
    rho3  = rho1 +  rho2
    xsi3  = xsi1 +  xsi2
    b3    = b1 +    b2
    zeta3 = zeta1 + zeta2               
    start = time.time()           
    if X.planes == 9:
        print 'Full polarimetry'  
        det1 = k1*xsi1*zeta1 + 2*real(a1*b1*conj(rho1)) - xsi1*(abs(rho1)**2) - k1*(abs(b1)**2) - zeta1*(abs(a1)**2)    
        det2 = k2*xsi2*zeta2 + 2*real(a2*b2*conj(rho2)) - xsi2*(abs(rho2)**2) - k2*(abs(b2)**2) - zeta2*(abs(a2)**2)       
        det3 = k3*xsi3*zeta3 + 2*real(a3*b3*conj(rho3)) - xsi3*(abs(rho3)**2) - k3*(abs(b3)**2) - zeta3*(abs(a3)**2)   
        p = 3
        f = p**2
        cst = p*((n+m)*log(n+m)-n*log(n)-m*log(m)) 
        rho = 1. - (2.*p**2-1.)*(1./n + 1./m - 1./(n+m))/(6.*p)    
        omega2 = -(p*p/4.)*(1. - 1./rho)**2 + p**2*(p**2-1.)*(1./n**2 + 1./m**2 - 1./(n+m)**2)/(24.*rho**2)                             
    elif X.planes == 5:
        print 'Full polarimetry, azimuthal symmetry'
        det1 = k1*xsi1*zeta1 - xsi1*(abs(rho1)**2)
        det2 = k2*xsi2*zeta2 - xsi2*(abs(rho2)**2)
        det3 = k3*xsi3*zeta3 - xsi3*(abs(rho3)**2)   
        p = 3  
        cst = p*((n+m)*log(n+m)-n*log(n)-m*log(m))    
        p = 2
        f1 = p**2
        rho1 = 1.-(2.*f1-1.)*(1./n+1./m-1./(n+m))/(6.*p)
        p = 1
        f2 = p**2
        rho2 = 1.-(2.*f2-1.)*(1./n+1./m-1./(n+m))/(6.*p)
        f = f1 + f2
        rho = (f1*rho1 + f2*rho2)/f
        omega2 = -f*(1.-1./rho)**2/4. + (f1*(f1-1.)+f2*(f2-1.))*(1./n**2+1./m**2-1./(n+m)**2)/(24.*rho**2)                          
    elif X.planes == 3:
        print 'Full polarimetry, diagonal only'
        det1 = k1*xsi1*zeta1 
        det2 = k2*xsi2*zeta2 
        det3 = k3*xsi3*zeta3      
        p = 3
        cst = p*((n+m)*log(n+m)-n*log(n)-m*log(m)) 
        p = 1
        f1 = p**2
        rho1 = 1-(2*f1-1)*(1./n+1./m-1./(n+m))/(6.*p)
        p = 1
        f2 = p**2
        rho2 = 1-(2*f2-1)*(1./n+1/m-1./(n+m))/(6.*p)
        p = 1
        f3 = p**2
        rho3 = 1-(2*f3-1)*(1./n+1/m-1./(n+m))/(6.*p)
        f = f1 + f2 + f3
        rho = (f1*rho1 + f2*rho2 + f3*rho3)/f
        omega2 = -f*(1-1./rho)**2/4. + (f1*(f1-1)+f2*(f2-1)+f3*(f3-1))*(1./n**2+1./m**2-1./(n+m)**2)/(24.*rho**2)                        
    elif X.planes == 4:
        print 'Dual polarimetry'
        det1 = k1*xsi1 - abs(a1)**2
        det2 = k2*xsi2 - abs(a2)**2 
        det3 = k3*xsi3 - abs(a3)**2        
        p = 2 
        cst = p*((n+m)*log(n+m)-n*log(n)-m*log(m)) 
        f = p**2
        rho = 1-(2*f-1)*(1./n+1./m-1./(n+m))/(6.*p)
        omega2 = -f/4.*(1-1./rho)**2 + f*(f-1)*(1./n**2+1./m**2-1./(n+m)**2)/(24.*rho**2)                         
    elif X.planes == 2:
        print 'Dual polarimetry, diagonal only'
        det1 = k1*xsi1
        det2 = k2*xsi2
        det3 = k3*xsi3  
        p = 2 
        cst = p*((n+m)*log(n+m)-n*log(n)-m*log(m))   
        p = 1
        f1 = p**2
        rho1 = 1-(2.*f1-1)*(1./n+1./m-1./(n+m))/(6.*p)
        p = 1
        f2 = p**2
        rho2 = 1-(2.*f2-1)*(1./n+1./m-1./(n+m))/(6.*p)
        f = f1 + f2
        rho = (f1*rho1 + f2*rho2)/f
        omega2 = -f*(1-1./rho)**2/4. + (f1*(f1-1)+f2*(f2-1))*(1./n**2+1./m**2-1./(n+m)**2)/(24.*rho**2)                         
    elif X.planes == 1:
        print 'One channel data only'
        det1 = k1 
        det2 = k2
        det3 = k3    
        p = 1 
        cst = p*((n+m)*log(n+m)-n*log(n)-m*log(m)) 
        f = p**2
        rho = 1-(2.*f-1)*(1./n+1./m-1./(n+m))/(6.*p)
        omega2 = -f/4.*(1-1./rho)**2+f*(f-1)*(1./n**2+1./m**2-1./(n+m)**2)/(24.*rho**2)                     
    else:
        print 'Incorrect input data'
        return 
    lnQ = cst+n*log(det1)+m*log(det2)-(n+m)*log(det3)
#  test statistic    
    Z = -2*rho*lnQ
#  change probabilty
    P =  (1.-omega2)*stats.chi2.cdf(Z,[f])+omega2*stats.chi2.cdf(Z,[f+4])
#  write to file system        
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,X.samples,Y.lines,2,GDT_Float32)
    if X.geotransform is not None:
        outDataset.SetGeoTransform(X.geotransform)
    if X.projection is not None:
        outDataset.SetProjection(X.projection) 
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