#!/usr/bin/env python
#******************************************************************************
#  Name:     ffncg.py
#  Purpose:  
#        Neural network image classification trained with scaled congugate gradient
#        with 10-fold cross-validation on multyvac
#  Note: 
#        Presently multyvac does not allow parallel processing. This script illustrates
#        the use of multyvac service. The map(crossvalidate,traintest) call in traintest()
#        will eventually (I hope) be parallelized.     
#  Usage:             
#    python ffncg.py
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
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
from shapely.geometry import asPolygon, MultiPoint
import matplotlib.pyplot as plt 
import shapely.wkt 
import numpy as np  
import gdal, ogr, osr, os, time
import multyvac as mv 

epochs = 500
 
def crossvalidate((Gstrn,lstrn,Gstst,lstst,L)):
    affn = Ffncg(Gstrn,lstrn,L)
    if affn.train(epochs=epochs):
        return affn.test(Gstst,lstst)
    else:
        return None

def traintst(Gs,ls,L): 
    m = np.shape(Gs)[0]
    traintest = []
    for i in range(10):
        sl = slice(i*m//10,(i+1)*m//10)
        traintest.append( (np.delete(Gs,sl,0), \
        np.delete(ls,sl,0),Gs[sl,:],ls[sl,:],L) )
    result = map(crossvalidate,traintest) 
    return result   

class Ffn(object):
    '''Abstract base class for 2-layer, 
       feed forward neural network'''     
     
    def __init__(self,Gs,ls,L): 
#      setup the network architecture        
        self._L=L 
        self._m,self._N = Gs.shape 
        self._K = ls.shape[1]
#      biased input as column vectors         
        Gs = np.mat(Gs).T 
        self._Gs = np.vstack((np.ones(self._m),Gs))      
#      biased output vector from hidden layer        
        self._n = np.mat(np.zeros(L+1))         
#      labels as column vectors
        self._ls = np.mat(ls).T
#      weight matrices
        self._Wh=np.mat(np.random. \
                        random((self._N+1,L)))-0.5
        self._Wo=np.mat(np.random. \
                        random((L+1,self._K)))-0.5             
    def forwardpass(self,G):
#      forward pass through the network
        expnt = self._Wh.T*G
        self._n = np.vstack((np.ones(1),1.0/ \
                                  (1+np.exp(-expnt))))
#      softwmax activation
        I = self._Wo.T*self._n
        A = np.exp(I-max(I))
        return A/np.sum(A)
    
    def classify(self,Gs):  
#      vectorized classes and membership probabilities
        Gs = np.mat(Gs).T
        m = Gs.shape[1]
        Gs = np.vstack((np.ones(m),Gs))
        expnt = self._Wh.T*Gs
        expnt[np.where(expnt<-100.0)] = -100.0
        expnt[np.where(expnt>100.0)] = 100.0        
        n=np.vstack((np.ones(m),1/(1+np.exp(-expnt))))
        Io = self._Wo.T*n
        maxIo = np.max(Io,axis=0)
        for k in range(self._K):
            Io[k,:] -= maxIo
        A = np.exp(Io)
        sm = np.sum(A,axis=0)
        Ms = np.zeros((self._K,m))
        for k in range(self._K):
            Ms[k,:] = A[k,:]/sm
        classes = np.argmax(Ms,axis=0)+1 
        return (classes, Ms)   
    
    def vforwardpass(self,Gs):
#      vectorized forward pass, Gs are biased column vectors
        m = Gs.shape[1]
        expnt = self._Wh.T*Gs
        n = np.vstack((np.ones(m),1.0/(1+np.exp(-expnt))))
        Io = self._Wo.T*n
        maxIo = np.max(Io,axis=0)
        for k in range(self._K):
            Io[k,:] -= maxIo
        A = np.exp(Io)
        sm = np.sum(A,axis=0) 
        Ms = np.zeros((self._K,m)) 
        for k in range(self._K):
            Ms[k,:] = A[k,:]/sm
        return (Ms, n)        
    
    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(ls.T,axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)
                       
    def cost(self):
        Ms, _ = self.vforwardpass(self._Gs)
        return -np.sum(np.multiply(self._ls,np.log(Ms+1e-20)))
    
class Ffncg(Ffn):
    
    def __init__(self,Gs,ls,L):
        Ffn.__init__(self,Gs,ls,L)
    
    def gradient(self):
#      gradient of cross entropy wrt synaptic weights          
        M,n = self.vforwardpass(self._Gs)
        D_o = self._ls - M
        D_h = np.mat(n.A*(1-n.A)*(self._Wo*D_o).A)[1::,:]
        dEh = -(self._Gs*D_h.T).ravel()
        dEo = -(n*D_o.T).ravel()
        return np.append(dEh.A,dEo.A)  
    
    def hessian(self):    
#      Hessian of cross entropy wrt synaptic weights        
        nw = self._L*(self._N+1)+self._K*(self._L+1)  
        v = np.eye(nw,dtype=np.float)  
        H = np.zeros((nw,nw))
        for i in range(nw):
            H[i,:] = self.rop(v[i,:])
        return H    
            
    def rop(self,V):     
#      reshape V to dimensions of Wh and Wo and transpose
        VhT = np.reshape(V[:(self._N+1)*self._L],(self._N+1,self._L)).T
        Vo = np.mat(np.reshape(V[self._L*(self._N+1)::],(self._L+1,self._K)))
        VoT = Vo.T
#      transpose the output weights
        Wo = self._Wo
        WoT = Wo.T 
#      forward pass
        M,n = self.vforwardpass(self._Gs) 
#      evaluation of v^T.H
        Z = np.zeros(self._m)  
        D_o = self._ls - M                          #d^o
        RIh = VhT*self._Gs                          #Rv{I^h}
        tmp = np.vstack((Z,RIh))                  
        RN = n.A*(1-n.A)*tmp.A                     #Rv{n}   
        RIo = WoT*RN + VoT*n                       #Rv{I^o}
        Rd_o = -np.mat(M*(1-M)*RIo.A)              #Rv{d^o}
        Rd_h = n.A*(1-n.A)*( (1-2*n.A)*tmp.A*(Wo*D_o).A + (Vo*D_o).A + (Wo*Rd_o).A )
        Rd_h = np.mat(Rd_h[1::,:])                          #Rv{d^h}
        REo = -(n*Rd_o.T - RN*D_o.T).ravel()       #Rv{dE/dWo}
        REh = -(self._Gs*Rd_h.T).ravel()            #Rv{dE/dWh}
        return np.hstack((REo,REh))                #v^T.H                                        
    
    def train(self,epochs=100):
        try: 
            cost = []             
            w = np.concatenate((self._Wh.A.ravel(),self._Wo.A.ravel()))
            nw = len(w)
            g = self.gradient()
            d = -g
            k = 0
            lam = 0.001
            while k < epochs:
                d2 = np.sum(d*d)                # d^2
                dTHd = np.sum(self.rop(d).A*d)  # d^T.H.d
                delta = dTHd + lam*d2
                if delta < 0:
                    lam = 2*(lam-delta/d2)
                    delta = -dTHd
                E1 = self.cost()                # E(w)
                dTg = np.sum(d*g)               # d^T.g
                alpha = -dTg/delta
                dw = alpha*d
                w += dw 
                self._Wh = np.mat(np.reshape(w[0:self._L*(self._N+1)],(self._N+1,self._L)))            
                self._Wo = np.mat(np.reshape(w[self._L*(self._N+1)::],(self._L+1,self._K)))
                E2 = self.cost()                # E(w+dw)
                Ddelta = -2*(E1-E2)/(alpha*dTg) # quadricity
                if Ddelta < 0.25:
                    w -= dw                     # undo weight change
                    self._Wh = np.mat(np.reshape(w[0:self._L*(self._N+1)],(self._N+1,self._L)))
                    self._Wo = np.mat(np.reshape(w[self._L*(self._N+1)::],(self._L+1,self._K)))     
                    lam *= 4.0                  # decrease step size
                    if lam > 1e20:              # if step too small
                        k = self.epochs         #     give up
                    else:                       # else
                        d = -g                  #     restart             
                else:
                    k += 1
                    cost.append(E1)             
                    if Ddelta > 0.75:
                        lam /= 2.0
                    g = self.gradient()
                    if k % nw == 0:
                        beta = 0.0
                    else:
                        beta = np.sum(self.rop(g).A*d)/dTHd
                    d = beta*d - g
            return cost 
        except Exception as e:
            print 'Error: %s'%e
            return None        

def main():  
    gdal.AllRegister()
    path = auxil.select_directory('Choose input directory')
    if path:
        os.chdir(path)        
#  input image    
    infile = auxil.select_infile(title='Choose image file') 
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
#  training data (shapefile)      
    trnfile = auxil.select_infile(filt='.shp',title='Choose train shapefile')
    if trnfile:
        trnDriver = ogr.GetDriverByName('ESRI Shapefile')
        trnDatasource = trnDriver.Open(trnfile,0)
        trnLayer = trnDatasource.GetLayer() 
        trnsr = trnLayer.GetSpatialRef()             
    else:
        return
#  hidden neurons
    L = auxil.select_integer(8,'number of hidden neurons')    
    if not L:
        return
#  outfile
    outfile, fmt = auxil.select_outfilefmt()   
    if not outfile:
        return     
#  coordinate transformation from training to image projection   
    ct= osr.CoordinateTransformation(trnsr,imsr) 
#  number of classes    
    feature = trnLayer.GetNextFeature() 
    while feature:
        classid = feature.GetField('CLASS_ID')
        feature = trnLayer.GetNextFeature() 
    trnLayer.ResetReading()    
    K = int(classid)+1       
    print '========================='
    print '       ffncg'
    print '========================='
    print time.asctime()    
    print 'image:    '+infile
    print 'training: '+trnfile          
#  loop through the polygons    
    Gs = [] # train observations
    ls = [] # class labels
    print 'reading training data...'
    for i in range(trnLayer.GetFeatureCount()):
        feature = trnLayer.GetFeature(i)
        classid = feature.GetField('CLASS_ID')
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
    m = len(ls)       
    print str(m) + ' training pixel vectors were read in' 
    Gs = np.array(Gs) 
    ls = np.array(ls)
#  stretch the pixel vectors to [-1,1]
    maxx = np.max(Gs,0)
    minx = np.min(Gs,0)
    for j in range(N):
        Gs[:,j] = 2*(Gs[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0 
#  random permutation of training data
    idx = np.random.permutation(m)
    Gs = Gs[idx,:] 
    ls = ls[idx,:]     
#  setup output dataset 
    driver = gdal.GetDriverByName(fmt)    
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Byte) 
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection) 
    outBand = outDataset.GetRasterBand(1) 
#  train on 9/10 training examples         
    Gstrn = Gs[0:9*m//10,:]
    lstrn = ls[0:9*m//10,:]
    affn = Ffncg(Gstrn,lstrn,L)
    print 'training on %i pixel vectors...' % np.shape(Gstrn)[0]
    start = time.time()
    cost = affn.train(epochs=epochs)
    print 'elapsed time %s' %str(time.time()-start) 
    if cost is not None:
#        cost = np.log10(cost)  
        ymax = np.max(cost)
        ymin = np.min(cost) 
        xmax = len(cost)      
        plt.plot(range(xmax),cost,'k')
        plt.axis([0,xmax,ymin-1,ymax])
        plt.title('Cross entropy')
        plt.xlabel('Epoch')              
#      classify the image           
        print 'classifying...'
        tile = np.zeros((cols,N))    
        for row in range(rows):
            for j in range(N):
                tile[:,j] = rasterBands[j].ReadAsArray(0,row,cols,1)
                tile[:,j] = 2*(tile[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0 
            cls, _ = affn.classify(tile)  
            outBand.WriteArray(np.reshape(cls,(1,cols)),0,row)
        outBand.FlushCache()
        outDataset = None
        inDataset = None  
        print 'thematic map written to: ' + outfile
        print 'please close the cross entropy plot to continue'
        plt.show()
    else:
        print 'an error occured' 
        return 
    
    print 'submitting cross-validation to multyvac'    
    start = time.time()
    jid = mv.submit(traintst,Gs,ls,L,_layer='ms_image_analysis')  
    print 'submission time: %s' %str(time.time()-start)
    start = time.time()    
    job = mv.get(jid)
    result = job.get_result(job) 
    
    
    print 'execution time: %s' %str(time.time()-start)      
    print 'misclassification rate: %f' %np.mean(result)
    print 'standard deviation:     %f' %np.std(result)         
    print '--------done---------------------'       

   
if __name__ == '__main__':
    main()