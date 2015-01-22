#!/usr/bin/env python
#******************************************************************************
#  Name:     classify.py
#  Purpose:  supervised classification of multispectral images
#  Usage:             
#    python classify.py
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
import auxil.header as header
import auxil.supervisedclass as sc
from auxil.auxil import ctable 
import gdal, ogr, osr, os, time
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
from shapely.geometry import asPolygon, MultiPoint
import matplotlib.pyplot as plt
import shapely.wkt  
import numpy as np
 
def main():      
    gdal.AllRegister()
    path = auxil.select_directory('Input directory')
    if path:
        os.chdir(path)        
#  input image    
    infile = auxil.select_infile(title='Image file') 
    if infile:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
        projection = inDataset.GetProjection()
        geotransform = inDataset.GetGeoTransform()
        if geotransform is not None:
            gt = list(geotransform) 
        else:
            print 'No geotransform available'
            return       
        imsr = osr.SpatialReference()  
        imsr.ImportFromWkt(projection)      
    else:
        return  
    pos =  auxil.select_pos(bands)   
    if not pos:
        return
    N = len(pos) 
    rasterBands = [] 
    for b in pos:
        rasterBands.append(inDataset.GetRasterBand(b)) 
#  training algorithm
    trainalg = auxil.select_integer(1,msg='1:Maxlike,2:Backprop,3:Congrad,4:SVM') 
    if not trainalg:
        return           
#  training data (shapefile)      
    trnfile = auxil.select_infile(filt='.shp',title='Train shapefile')
    if trnfile:
        trnDriver = ogr.GetDriverByName('ESRI Shapefile')
        trnDatasource = trnDriver.Open(trnfile,0)
        trnLayer = trnDatasource.GetLayer() 
        trnsr = trnLayer.GetSpatialRef()             
    else:
        return     
    tstfile = auxil.select_outfile(filt='.tst', title='Test results file') 
    if not tstfile:
        print 'No test output'      
#  outfile
    outfile, outfmt = auxil.select_outfilefmt(title='Classification file')   
    if not outfile:
        return                   
    if trainalg in (2,3,4):
#      class probabilities file, hidden neurons
        probfile, probfmt = auxil.select_outfilefmt(title='Probabilities file')
    else:
        probfile = None     
    if trainalg in (2,3):    
        L = auxil.select_integer(8,'Number of hidden neurons')    
        if not L:
            return                  
#  coordinate transformation from training to image projection   
    ct= osr.CoordinateTransformation(trnsr,imsr) 
#  number of classes    
    K = 1
    feature = trnLayer.GetNextFeature() 
    while feature:
        classid = feature.GetField('CLASS_ID')
        if int(classid)>K:
            K = int(classid)
        feature = trnLayer.GetNextFeature() 
    trnLayer.ResetReading()    
    K += 1       
    print '========================='
    print 'supervised classification'
    print '========================='
    print time.asctime()    
    print 'image:    '+infile
    print 'training: '+trnfile  
    if trainalg == 1:
        print 'Maximum Likelihood'
    elif trainalg == 2:
        print 'Neural Net (Backprop)'
    elif trainalg ==3:
        print 'Neural Net (Congrad)'
    else:
        print 'Support Vector Machine'               
#  loop through the polygons    
    Gs = [] # train observations
    ls = [] # class labels
    classnames = '{unclassified'
    classids = set()
    print 'reading training data...'
    for i in range(trnLayer.GetFeatureCount()):
        feature = trnLayer.GetFeature(i)
        classid = str(feature.GetField('CLASS_ID'))
        classname  = feature.GetField('CLASS_NAME')
        if classid not in classids:
            classnames += ',   '+ classname
        classids = classids | set(classid)        
        l = [0 for i in range(K)]
        l[int(classid)] = 1.0
        polygon = feature.GetGeometryRef()
#      transform to same projection as image        
        polygon.Transform(ct)  
#      convert to a Shapely object            
        poly = shapely.wkt.loads(polygon.ExportToWkt())
#      transform the boundary to pixel coords in numpy        
        bdry = np.array(poly.boundary) 
        bdry[:,0] = bdry[:,0]-gt[0]
        bdry[:,1] = bdry[:,1]-gt[3]
        GT = np.mat([[gt[1],gt[2]],[gt[4],gt[5]]])
        bdry = bdry*np.linalg.inv(GT) 
#      polygon in pixel coords        
        polygon1 = asPolygon(bdry)
#      raster over the bounding rectangle        
        minx,miny,maxx,maxy = map(int,list(polygon1.bounds))  
        pts = [] 
        for i in range(minx,maxx+1):
            for j in range(miny,maxy+1): 
                pts.append((i,j))             
        multipt =  MultiPoint(pts)   
#      intersection as list              
        intersection = np.array(multipt.intersection(polygon1),dtype=np.int).tolist()
#      cut out the bounded image cube               
        cube = np.zeros((maxy-miny+1,maxx-minx+1,len(rasterBands)))
        k=0
        for band in rasterBands:
            cube[:,:,k] = band.ReadAsArray(minx,miny,maxx-minx+1,maxy-miny+1)
            k += 1
#      get the training vectors
        for (x,y) in intersection:         
            Gs.append(cube[y-miny,x-minx,:])
            ls.append(l)   
        polygon = None
        polygon1 = None            
        feature.Destroy()  
    trnDatasource.Destroy() 
    classnames += '}'
    m = len(ls)       
    print str(m) + ' training pixel vectors were read in' 
    Gs = np.array(Gs) 
    ls = np.array(ls)
#  stretch the pixel vectors to [-1,1] for ffn
    maxx = np.max(Gs,0)
    minx = np.min(Gs,0)
    for j in range(N):
        Gs[:,j] = 2*(Gs[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0 
#  random permutation of training data
    idx = np.random.permutation(m)
    Gs = Gs[idx,:] 
    ls = ls[idx,:]     
#  setup output datasets 
    driver = gdal.GetDriverByName(outfmt)    
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Byte) 
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection) 
    outBand = outDataset.GetRasterBand(1) 
    if probfile:
        driver = gdal.GetDriverByName(probfmt)    
        probDataset = driver.Create(probfile,cols,rows,K,GDT_Byte) 
        if geotransform is not None:
            probDataset.SetGeoTransform(tuple(gt))
        if projection is not None:
            probDataset.SetProjection(projection)  
        probBands = [] 
        for k in range(K):
            probBands.append(probDataset.GetRasterBand(k+1))         
    if tstfile:
#  train on 2/3 training examples         
        Gstrn = Gs[0:2*m//3,:]
        lstrn = ls[0:2*m//3,:] 
        Gstst = Gs[2*m//3:,:]  
        lstst = ls[2*m//3:,:]    
    else:
        Gstrn = Gs
        lstrn = ls         
    if   trainalg == 1:
        classifier = sc.Maxlike(Gstrn,lstrn)
    elif trainalg == 2:
        classifier = sc.Ffnbp(Gstrn,lstrn,L)
    elif trainalg == 3:
        classifier = sc.Ffncg(Gstrn,lstrn,L)
    elif trainalg == 4:
        classifier = sc.Svm(Gstrn,lstrn)         
            
    print 'training on %i pixel vectors...' % np.shape(Gstrn)[0]
    start = time.time()
    result = classifier.train()
    print 'elapsed time %s' %str(time.time()-start) 
    if result:
        if trainalg in [2,3]:
            cost = np.log10(result)  
            ymax = np.max(cost)
            ymin = np.min(cost) 
            xmax = len(cost)      
            plt.plot(range(xmax),cost,'k')
            plt.axis([0,xmax,ymin-1,ymax])
            plt.title('Log(Cross entropy)')
            plt.xlabel('Epoch')              
#      classify the image           
        print 'classifying...'
        start = time.time()
        tile = np.zeros((cols,N))    
        for row in range(rows):
            for j in range(N):
                tile[:,j] = rasterBands[j].ReadAsArray(0,row,cols,1)
                tile[:,j] = 2*(tile[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0               
            cls, Ms = classifier.classify(tile)  
            outBand.WriteArray(np.reshape(cls,(1,cols)),0,row)
            if probfile:
                Ms = np.byte(Ms*255)
                for k in range(K):
                    probBands[k].WriteArray(np.reshape(Ms[k,:],(1,cols)),0,row)
        outBand.FlushCache()
        print 'elapsed time %s' %str(time.time()-start)
        outDataset = None
        inDataset = None      
        if probfile:
            for probBand in probBands:
                probBand.FlushCache() 
            probDataset = None
            print 'class probabilities written to: %s'%probfile   
        K =  lstrn.shape[1]+1                     
        if (outfmt == 'ENVI') and (K<19):
#          try to make an ENVI classification header file            
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
            hdr['classes'] = str(K)
            classlookup = '{0'
            for i in range(1,3*K):
                classlookup += ', '+str(str(ctable[i]))
            classlookup +='}'    
            hdr['class lookup'] = classlookup
            hdr['class names'] = classnames
            f = open(headerfile,'w')
            f.write(str(hdr))
            f.close()             
        print 'thematic map written to: %s'%outfile
        if trainalg in [2,3]:
            print 'please close the cross entropy plot to continue'
            plt.show()
        if tstfile:
            with open(tstfile,'w') as f:
                print >>f, 'FFN test results for %s'%infile
                print >>f, time.asctime()
                print >>f, 'Classification image: %s'%outfile
                print >>f, 'Class probabilities image: %s'%probfile
                print >>f, lstst.shape[0],lstst.shape[1]
                classes, _ = classifier.classify(Gstst)
                labels = np.argmax(lstst,axis=1)+1
                for i in range(len(classes)):
                    print >>f, classes[i], labels[i]              
                f.close()
                print 'test results written to: %s'%tstfile
        print 'done'
    else:
        print 'an error occured' 
        return 
   
if __name__ == '__main__':
    main()