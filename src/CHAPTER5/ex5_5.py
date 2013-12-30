#!/usr/bin/env python
#Name:  ex5_5.py
import auxil.auxil as auxil
import os
from numpy import *
from osgeo import gdal
from scipy import ndimage
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte

def parse_gcp(gcpfile):
    with open(gcpfile) as f:
        pts = []
        for i in range(6):
            line =f.readline()
        while line:
            pts.append(map(eval,line.split()))
            line = f.readline()
        f.close()    
    pts = array(pts)     
    return (pts[:,:2],pts[:,2:])  
    
def main(): 
    gdal.AllRegister()
    path = auxil.select_directory('Working directory')
    if path:
        os.chdir(path)        
    file1=auxil.select_infile(title='Base image') 
    if file1:                   
        inDataset1 = gdal.Open(file1,GA_ReadOnly)     
        cols1 = inDataset1.RasterXSize
        rows1 = inDataset1.RasterYSize
        print 'Base image: %s'%file1    
    else:
        return     
    file2=auxil.select_infile(title='Warp image') 
    if file2:                  
        inDataset2 = gdal.Open(file2,GA_ReadOnly)     
        cols2 = inDataset2.RasterXSize
        rows2 = inDataset2.RasterYSize
        bands2 = inDataset2.RasterCount        
        print 'Warp image: %s'%file2    
    else:
        return 
    file3 = auxil.select_infile(title='GCP file',\
                                  filt='pts')  
    if file3:
        pts1,pts2 = parse_gcp(file3)
    else:
        return
    outfile,fmt = auxil.select_outfilefmt() 
    if not outfile:
        return   
    image2 = zeros((bands2,rows2,cols2))                                   
    for k in range(bands2):
        band = inDataset2.GetRasterBand(k+1)
        image2[k,:,:]=band.ReadAsArray(0,0,cols2,rows2)
    inDataset2 = None
    n = len(pts1)    
    y = pts1.ravel()
    A = zeros((2*n,4))
    for i in range(n):
        A[2*i,:] =   [pts2[i,0],-pts2[i,1],1,0]
        A[2*i+1,:] = [pts2[i,1], pts2[i,0],0,1]   
    a,b,x0,y0 = linalg.lstsq(A,y)[0]
    R = array([[a,-b],[b,a]])     
    warped = zeros((bands2,rows1,cols1),dtype=uint8) 
    for k in range(bands2):
        tmp = ndimage.affine_transform(image2[k,:,:],R)
        warped[k,:,:]=tmp[-y0:-y0+rows1,-x0:-x0+cols1]   
    driver = gdal.GetDriverByName(fmt)   
    outDataset = driver.Create(outfile,
                    cols1,rows1,bands2,GDT_Byte)    
    geotransform = inDataset1.GetGeoTransform()
    projection = inDataset1.GetProjection()   
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection)        
    for k in range(bands2):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(warped[k,:,:],0,0) 
        outBand.FlushCache()
    outDataset = None
    inDataset1 = None       
    print 'Warped image written to: %s'%outfile        

if __name__ == '__main__':
    main()    