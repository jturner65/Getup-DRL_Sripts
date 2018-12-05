#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 08:27:24 2018

@author: john
"""

import numpy as np
import scipy as scp
import scipy.stats as stats
import matplotlib.pyplot as plt
#test formulations of reward functions

def calcActionDiffRew(a, wt=1.0):
    adt = a.dot(a)
    rew = wt * np.exp(-adt)
    return rew, adt

#penalty/reward == 0 at tarx, small positive value <tarx and large negative value > tarx
def calcExpDiffRwd(newx, tarx, scl, yOffset=0, wt=1.0):
    val = ((newx - tarx)/scl)
    rew = wt * (np.exp(val)-yOffset)
    return rew, val

#        wt = self.rwdWts[typ] 
#        var = self.rwdVars[typ]
#        tol = self.rwdTols[typ]
#        rew = wt * (np.exp((chkVal-tol)/var)-offset)



#build exponential reward function based on height difference between COM and foot center
def calcHeightDiffRew(heightDiff, maxHt, rwdMult=10, pwr=1.2):
    ratio = (heightDiff/maxHt)#(maxHt/heightDiff)
    if(heightDiff < 0):
        #can't be negative in exponential, or yields complex number result
        #retain sign to preserve negative value
        ratio *=-1
        height_rew = -(ratio**pwr)#20.0 * heightAfter
    else:
        height_rew = ratio**pwr#20.0 * heightAfter
    return height_rew *rwdMult

def calcHeightDiffRewNew(heightDiff, rwdMult=10):
    #curve to benefit early raising more over linear and increasingly benefit approaching max height with higher slope than linear
    #[calcHeightDiffRew(x,maxHt, rwdMult=10, pwr=5.0) + calcHeightDiffRew(x,maxHt, rwdMult=10, pwr=1.0) + calcHeightDiffRew(x,maxHt/2.0, rwdMult=10, pwr=0.4) for x in testX]
    ratio = (heightDiff/0.87790457356624751)
    rwdSignMult = 1.0
    if(heightDiff < 0):
        #can't be negative in exponential, or yields complex number result
        #retain sign to preserve negative value
        ratio *=-1
        rwdSignMult *= -1.0
    
    height_rew = rwdSignMult * ((ratio**5) + ratio + (2.0*ratio)**0.4)
    return height_rew *rwdMult


def calcVelRwd(vel, vMaxRwd, minVel, maxRwd):
    #this will retain maximum == vMaxRwd ->parabola is tighter
    denom = ((vMaxRwd-minVel) * (vMaxRwd-minVel))
    a = maxRwd if denom == 0 else maxRwd/denom
    cval = (vel-vMaxRwd) 
#        #below shifts whole parabola and so changes x value that gives max reward
#        a = maxRwd/(vMaxRwd * vMaxRwd)
#        cval = (vel-(vMaxRwd + minVel))        
    return (-a *(cval * cval) + maxRwd)

#this will yield the reward parabola given min and 
#max vals that yield rewards (roots of parabola) and reward for peak
def calcVelRwdRootMethod(val, minPosVal, maxPosVal, maxRwd):
    xMax = (minPosVal + maxPosVal)/2.0
    mult = maxRwd/((xMax-minPosVal) * (xMax - maxPosVal))    
    return mult * (val - minPosVal) * (val - maxPosVal)
    


def calcVelRwdOld(vel, vMaxRwd, minVel, maxRwd):
    a = maxRwd/((vMaxRwd-minVel) * (vMaxRwd-minVel))
    cval = (vel-(vMaxRwd))# + minVel))        
    return (-a *(cval * cval) + maxRwd)

def calcVelRwdOld2(vel, vMaxRwd, minVel, maxRwd):
    a = maxRwd/(vMaxRwd * vMaxRwd)
    cval = (vel-(vMaxRwd + minVel))        
    return (-a *(cval * cval) + maxRwd)
   
#raiseVelScore = -1/valY*(raiseVel-(valY+ self.minUpVel))**2 + valY + self.minUpVel
def ak_calcVelRwdUp(velCheck, valMult, minVel):
    arg = -1/valMult
    vel = (velCheck - (valMult + minVel))
    y = arg*(vel)**2+valMult + minVel   
    dy = 2*arg*vel
    return y,dy

def ak_calcVelRwdX(vel, valX):
    arg = -1/valX
    y = arg*(vel)**2+valX    
    dy = 2*arg*vel
    return y,dy


def runTestsAndPlot(maxX, minVelAllowed, maxRwd):
    x = np.linspace(maxX-1,maxX+1,num=110)
    x1 = np.linspace(maxX-minVelAllowed,maxX+minVelAllowed,num=11)    
    y = calcVelRwd(x, maxX, minVelAllowed , maxRwd)
    yz = calcVelRwd(0, maxX, minVelAllowed , maxRwd)
    ymax1 = calcVelRwd(maxX, maxX, minVelAllowed , maxRwd)
    y1=calcVelRwd(x1, maxX, minVelAllowed , maxRwd)
    ymin1 = calcVelRwd(minVelAllowed, maxX, minVelAllowed , maxRwd)
    yx1 = calcVelRwd(5, maxX, minVelAllowed , maxRwd)
    plt.plot(x,y)
    plt.plot(x1,y1)

def mmntAnalyze(adota):
    minAdot = np.min(adota)
    maxAdot = np.max(adota)
    meanAdot = np.mean(adota)
    stdAdot = np.std(adota)
    skewAdot = stats.skew(adota)
    kurtAdot = stats.kurtosis(adota)
    tmpData = np.random.normal(meanAdot,stdAdot)
    plt.hist(adota, bins='auto')
    #plt.hist(tmpData, bins='auto')    

    
#x at peak
maxX = 4
#minimum allowed x value to receive a positive reward
minVelAllowed = .5
#max y value - value at maxX
maxRwd = 4.0


#runTestsAndPlot(maxX, minVelAllowed, maxRwd)
#start at 0 (ht above start seated COM ht), desired end is  0.87790457356624751 (== standing ht above avg foot loc with both feet on ground and legs straight )
htTestX = np.linspace(0, 0.878,num=200)
pwrTest = np.linspace(1.0,5.0,num=13)
clrMapFunc = plt.cm.get_cmap('hsv', len(pwrTest)+1)
maxHt = 0.87790457356624751
testX = htTestX
for i in range(len(pwrTest)):
    yHtTest = [calcHeightDiffRew(x,maxHt, pwr=pwrTest[i]) + calcHeightDiffRew(x,maxHt, pwr=(1.0/pwrTest[i])) for x in testX]
    plt.plot(testX,yHtTest, color=clrMapFunc(i))
plt.legend(pwrTest)

yHtTest = [calcHeightDiffRew(x,maxHt, rwdMult=10, pwr=5.0) + calcHeightDiffRew(x,maxHt, rwdMult=10, pwr=1.0) + calcHeightDiffRew(x,maxHt/2.0, rwdMult=10, pwr=0.4) for x in testX]
plt.plot(testX,yHtTest, color=clrMapFunc(0))
yHtTestl = [3*calcHeightDiffRew(x,maxHt, rwdMult=10, pwr=1.0) for x in testX]
plt.plot(testX,yHtTestl, color=clrMapFunc(5))


yHtTest = [calcHeightDiffRewNew(x)for x in testX]
plt.plot(testX,yHtTest, color=clrMapFunc(3))
yHtTestl = [3*calcHeightDiffRew(x,maxHt, rwdMult=10, pwr=1.0) for x in testX]
plt.plot(testX,yHtTestl, color=clrMapFunc(5))

#####
#test exponential penalty
expXtest = np.linspace(0, 5,num=201)
scls = np.linspace(0.1,1.1, num=11)
y=np.zeros([expXtest.shape[0],scls.shape[0]])
for i in range(len(expXtest)):
    x = expXtest[i]
    dbgStr = 'x : {:.3f}'.format(x)
    for j in range(len(scls)):
        y[i][j],val = calcExpDiffRwd(x,0.1, scls[j],yOffset=0,wt=1.0)
        dbgStr += '|s:{:.3f}|r:{:.3f}'.format(scls[j], y[i][j])

    print(dbgStr)


plt.ylim(ymax=20, ymin=-20)
plt.plot(expXtest, y[:])
plt.legend(scls)

#test action vector reward
#show dist of rewards
numVals = 1000000
vSize = 23
x=np.linspace(1,numVals, num=numVals)
y=np.zeros(numVals)
adota=np.zeros(numVals)
for i in range(numVals):
    aVec = np.random.uniform(low=-1, high=1, size=vSize)
    y[i], adota[i]=calcActionDiffRew(aVec)
    
#plt.plot(x, y)
#plt.plot(x, adota)
    
#mean value of a.dot(a) = x/3, where x is dimension of a
#mmntAnalyze(y)

xExp = np.linspace(0, vSize, numVals)
expAdotaSqrt_yVals = np.exp(-1*(xExp/np.sqrt(vSize)))
expAdotaLog_yVals = np.exp(-1*(xExp/np.log(vSize)))
oneOvAdota_yVals = 1.0/(1+xExp)
plt.plot(xExp, expAdotaSqrt_yVals, label='exp(- a.dot(a)/sqrt(sz))')
plt.plot(xExp, expAdotaLog_yVals, label='exp(- a.dot(a)/log(sz))')
plt.plot(xExp, oneOvAdota_yVals, label='1/(1.0+a.dot(a))')
plt.legend()



minRoot = minVelAllowed
x_peakToRoot = maxX - minRoot
maxRoot = maxX + x_peakToRoot
x = np.linspace(minRoot-x_peakToRoot,maxRoot+x_peakToRoot,num=110)
x1 = np.linspace(minRoot-1,maxRoot+1,num=11)    

y = calcVelRwdRootMethod(x, minRoot, maxRoot, maxRwd)
y1 = calcVelRwdRootMethod(x1, minRoot, maxRoot, maxRwd)
plt.plot(x,y)
plt.plot(x1,y1)











#yold = calcVelRwdOld(x, maxX, minVelAllowed , maxRwd)
#ymax2 = calcVelRwdOld(maxX+minVelAllowed, maxX, minVelAllowed , maxRwd)
#ymin2 = calcVelRwdOld(minVelAllowed, maxX, minVelAllowed , maxRwd)
#yx2 = calcVelRwdOld(5, maxX, minVelAllowed , maxRwd)
#plt.plot(x,yold)

yak, dyak = ak_calcVelRwdX(x, 1.5)
ymax, dymax = ak_calcVelRwdX(x1, 1.5)

#for side vel
x_ak = np.linspace(-5,+5,num=200)
x_akMax = np.linspace(-.1,.1,num=7)
yakside, dyakside = ak_calcVelRwdX(x_ak, 1)
ymaxside, dymaxside = ak_calcVelRwdX(x_akMax, 1)
plt.plot(x_ak,yakside)
plt.plot(x_akMax,ymaxside)

vm =1.75/2
x2 = np.linspace(vm-.01,vm+.01,num=7)
yakup, dyakup = ak_calcVelRwdUp(x, vm,.001)
ymaxup, dymaxup = ak_calcVelRwdUp(x2, vm,.001)
#print(yz)
#plt.plot(x,y)
#plt.plot(x,yak)
plt.plot(x,yakup)
plt.plot(x,dyakup)

#plt.plot(x,dyak)

max(yakup)


#
#    
#    #        velScore = -(vel)**2 + 1.0
#        #medium
#        valX = 1.5
#        velScore = -1/valX*(vel)**2+valX