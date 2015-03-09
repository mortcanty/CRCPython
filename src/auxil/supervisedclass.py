#!/usr/bin/env python
#******************************************************************************
#  Name:     supervisedclass.py
#  Purpose:  object classes for supervised image classification, maximum likelihood, 
#            back-propagation, scaled conjugate gradient, support vector machine
#  Usage:    
#     import supervisedclass
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


import numpy as np  
from mlpy import MaximumLikelihoodC, LibSvm    

class Maxlike(MaximumLikelihoodC):
     
    def __init__(self,Gs,ls): 
        MaximumLikelihoodC.__init__(self)
        self._K = ls.shape[1] 
        self._Gs = Gs 
        self._N = Gs.shape[1]
        self._ls = ls
        
    def train(self):
        try: 
            labels = np.argmax(self._ls,axis=1)
            idx = np.where(labels == 0)[0]
            ls = np.ones(len(idx),dtype=np.int)
            Gs = self._Gs[idx,:]
            for k in range(1,self._K):
                idx = np.where(labels == k)[0]
                ls = np.concatenate((ls, \
                (k+1)*np.ones(len(idx),dtype=np.int)))
                Gs = np.concatenate((Gs,\
                               self._Gs[idx,:]),axis=0)         
            self.learn(Gs,ls)    
            return True 
        except Exception as e:
            print 'Error: %s'%e 
            return False    
            
    def classify(self,Gs):
        classes = self.pred(Gs)
        return (classes, None)    
    
    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)
        

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
        labels = np.argmax(ls.T,axis=0) + 1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)
                       
    def cost(self):
        Ms, _ = self.vforwardpass(self._Gs)
        return -np.sum(np.multiply(self._ls,np.log(Ms+1e-20)))
    
    
class Ffnbp(Ffn):
    
    def __init__(self,Gs,ls,L,epochs=1000):
        Ffn.__init__(self,Gs,ls,L)
        self.epochs=epochs
           
    def train(self):
        eta = 0.01
        alpha = 0.5
        maxitr = self.epochs*self._m 
        inc_o1 = 0.0
        inc_h1 = 0.0
        epoch = 0
        cost = []
        itr = 0        
        try:
            while itr<maxitr:
#              select example pair at random
                nu = np.random.randint(0,self._m)
                x = self._Gs[:,nu]
                ell = self._ls[:,nu]
#              send it through the network
                m = self.forwardpass(x)
#              determine the deltas
                d_o = ell - m
                d_h = np.multiply(np.multiply(self._n,\
                     (1-self._n)),(self._Wo*d_o))[1::]
#              update synaptic weights
                inc_o = eta*(self._n*d_o.T)
                inc_h = eta*(x*d_h.T)
                self._Wo += inc_o + alpha*inc_o1
                self._Wh += inc_h + alpha*inc_h1
                inc_o1 = inc_o
                inc_h1 = inc_h
#              record cost function
                if itr % self._m == 0:
                    cost.append(self.cost())
                    epoch += 1
                itr += 1
        except Exception as e:
            print 'Error: %s'%e
            return None
        return np.array(cost)
    
class Ffncg(Ffn):
    
    def __init__(self,Gs,ls,L,epochs=1000):
        Ffn.__init__(self,Gs,ls,L)
        self.epochs=epochs
    
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
                                         
    
    def train(self):
        try: 
            cost = []             
            w = np.concatenate((self._Wh.A.ravel(),self._Wo.A.ravel()))
            nw = len(w)
            g = self.gradient()
            d = -g
            k = 0
            lam = 0.001
            while k < self.epochs:
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

    
class Svm(object):   
    
    def __init__(self,Gs,ls):
        self._K = ls.shape[1]
        self._Gs = Gs
        self._N = Gs.shape[1]
        self._ls = ls
        self._svm = LibSvm('c_svc','rbf',\
            gamma=1.0/self._N,C=100,probability=True)                
    
    def train(self):
        try:
            labels = np.argmax(self._ls,axis=1)
            idx = np.where(labels == 0)[0]
            ls = np.ones(len(idx),dtype=np.int)
            Gs = self._Gs[idx,:]
            for k in range(1,self._K):
                idx = np.where(labels == k)[0]
                ls = np.concatenate((ls, \
                (k+1)*np.ones(len(idx),dtype=np.int)))
                Gs = np.concatenate((Gs,\
                               self._Gs[idx,:]),axis=0)         
            self._svm.learn(Gs,ls)  
            return True 
        except Exception as e:
            print 'Error: %s'%e 
            return False    
    
    def classify(self,Gs):
        probs = np.transpose(self._svm. \
                             pred_probability(Gs))       
        classes = np.argmax(probs,axis=0)+1
        return (classes, probs) 
    
    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)
    
if __name__ == '__main__':
#  test on random data    
    Gs = 2*np.random.random((100,3)) -1.0
    ls = np.zeros((100,6))
    for l in ls:
        l[np.random.randint(0,6)]=1.0 
    cl = Ffnbp(Gs,ls,4)  
    if cl.train() is not None:
        classes, _ = cl.classify(Gs) 
    print classes