#!/usr/bin/env python
#Name:  ex1_3.py
import auxil.auxil as auxil
from numpy import *
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32

def main(): 
    gdal.AllRegister()
    infile = auxil.select_infile() 
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    else:
        return
    
#  spectral and spatial subsets    
    pos =  auxil.select_pos(bands)
    bands = len(pos)    
    x0,y0,rows,cols=auxil.select_dims([0,0,rows,cols])
    
#  data matrix
    G = zeros((rows*cols,len(pos))) 
    k = 0                                   
    for b in pos:
        band = inDataset.GetRasterBand(b)
        tmp = band.ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()
        G[:,k] = tmp - mean(tmp)
        k += 1
        
#  covariance matrix
    C = mat(G).T*mat(G)/(cols*rows-1)
    
#  diagonalize    
    lams,U = linalg.eigh(C)
     
#  sort
    idx = argsort(lams)[::-1]
    lams = lams[idx]
    U = U[:,idx]         
               
#  project
    PCs = reshape(array(G*U),(rows,cols,bands))   
    
#  write to disk       
    outfile,fmt = auxil.select_outfilefmt() 
    if outfile:
        driver = gdal.GetDriverByName(fmt)   
        outDataset = driver.Create(outfile,
                        cols,rows,bands,GDT_Float32)
        projection = inDataset.GetProjection()
        geotransform = inDataset.GetGeoTransform()
        if geotransform is not None:
            gt = list(geotransform)
            gt[0] = gt[0] + x0*gt[1]
            gt[3] = gt[3] + y0*gt[5]
            outDataset.SetGeoTransform(tuple(gt))
        if projection is not None:
            outDataset.SetProjection(projection)        
        for k in range(bands):        
            outBand = outDataset.GetRasterBand(k+1)
            outBand.WriteArray(PCs[:,:,k],0,0) 
            outBand.FlushCache() 
        outDataset = None    
    inDataset = None        
 
if __name__ == '__main__':
    main()    