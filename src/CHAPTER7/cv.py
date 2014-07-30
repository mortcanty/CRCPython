#!/usr/bin/env python
#******************************************************************************
#  Name:     cv.py
#  Purpose:  neural network image classification trained with scaled congugate gradient
#            with 10-fold cross-validation on picloud, called from IDL
#  Usage:             
#    python cv.py
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

import cloud, time, ffn 
import numpy as np
 
def crossvalidate((Gstrn,lstrn,Gstst,lstst,L)):
    affn = ffn.Ffncg(Gstrn,lstrn,L)
    if affn.train() is not None:
        return affn.test(Gstst,lstst)
    else:
        return None

def traintst(Gs,ls,L):
    m = np.shape(Gs)[0]
    traintest = []
    for i in range(10):
        sl = slice(i*m//10,(i+1)*m//10)
        traintest.append( (np.delete(Gs,sl,0),np.delete(ls,sl,0),Gs[sl,:],ls[sl,:],L) )
    jids = cloud.map(crossvalidate,traintest,_type='c1') 
    return cloud.result(jids)    

def main():
    cloud.setkey(2329,'270cb3cccb9beb65d2f424b24ccbd5a920c5ccef')   
    try:
        fn = raw_input()
        f = open(fn)
        L = float(f.readline())           
        line = f.readline()        
        data = []
        while line:
            d = map(eval,line.split())
            data.append(d)
            line = f.readline()         
        f.close()
        n = len(data)            
        Gs = np.array(data[0:n/2])
        ls = np.array(data[n/2::])
        outstr = ''
        outstr += 'submitting cross validation to picloud\n'
        cloud.config.max_transmit_data=12000000    
        start = time.time()
        jid = cloud.call(traintst,Gs,ls,L)  
        outstr += 'submission time: %s\n' %str(time.time()-start)
        start = time.time()    
        result = cloud.result(jid) 
        outstr += 'cloud execution time: %s\n' %str(time.time()-start)      
        outstr += 'misclassification rate: %f\n' %np.mean(result)
        outstr += 'standard deviation:     %f\n' %np.std(result)         
        outstr += '--------done---------------------'  
        print outstr
    except:
        print 'an error occurred'     

   
if __name__ == '__main__':
    main()