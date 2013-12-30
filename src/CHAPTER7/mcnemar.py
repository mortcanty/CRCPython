#!/usr/bin/env python
#******************************************************************************
#  Name:     mcnemar.py
#  Purpose:  compare two classifiers with McNemar statistic
#  Usage:             
#    python mcnenmar.py
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
import numpy as np
from scipy import stats

def main():
    tstfile1 = auxil.select_infile(filt='.tst', title='Test results file, first classifier') 
    if not tstfile1:
        return
    tstfile2 = auxil.select_infile(filt='.tst', title='Test results file, second classifier') 
    if not tstfile2:
        return    
    print '========================='
    print '     McNemar test'
    print '========================='
    with open(tstfile1,'r') as f1:
        with open(tstfile2,'r') as f2:
            line = ''
            for i in range(4):
                line += f1.readline()
            print 'first classifier:\n'+line    
            line = f1.readline().split()
            n1 = int(line[0]) 
            K1 = int(line[1]) 
            line = ''               
            for i in range(4):
                line += f2.readline()
            print 'second classifier:\n'+line   
            line = f2.readline().split()    
            n2 = int(line[0]) 
            K2 = int(line[1])
            if (n1 != n2) or (K1 != K2):
                print 'test files are incompatible'
                return
            print 'test observbations: %i'%n1
            print 'classes: %i'%K1
#          calculate McNemar
            y10 = 0.0
            y01 = 0.0
            for i in range(n1):
                line = f1.readline()
                k = map(int,line.split())
                k1A = k[0]
                k2A = k[1]
                line = f2.readline()
                k = map(int,line.split())
                k1B = k[0]
                k2B = k[1]
                if (k1A != k2A) and (k1B == k2B):
                    y10 += 1
                if (k1A == k2A) and (k1B != k2B):
                    y01 += 1        
    f1.close()
    f2.close()
    McN = (np.abs(y01-y10))**2/(y10+y01)
    print 'first classifier: %i'%int(y10)      
    print 'second classifier: %i'%int(y01)
    print 'McNemar statistic: %f'%McN
    print 'P-value: %f'%(1-stats.chi2.cdf(McN,1))
          
if __name__ == '__main__':
    main()     