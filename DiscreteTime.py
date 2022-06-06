import numpy as np
from numpy import int64
import matplotlib.pyplot as plt

class Signal:
  x = 0
  n = 0
  def __init__(self, xin, nStart, nEnd):
    self.x = np.array(xin)
    self.n = np.arange(nStart,nEnd+1,1)
  def plot(self):
    plt.stem(self.n, self.x)
    plt.xticks(self.n)
    plt.xlabel("n")
    plt.ylabel("values")
    plt.show()
  def flip(self):
    self.x = np.flip(self.x)
    self.n = -np.flip(self.n)
  def shift(self,n0):
    if n0 >= 0:
      self.x[n0:] = self.x[:-n0]
      self.x[:n0] = 0
    else:
      self.x[:n0] = self.x[-n0:]
      self.x[n0:] = 0
  def decimate(self, D):
    indShift = 0 - self.n[0]
    tempX = np.compress((self.n % D) == 0,self.x)
    tempN = np.compress((self.n % D) == 0,self.n)
    self.x = np.zeros((len(self.x)))
    for i in range(len(tempN)):
      for j in range(len(self.x)):
        if(j==tempN[i]+indShift):
          self.x[j] = tempX[i]
  def expand(self,U):
    tempX = np.zeros(U*len(self.x)-(U-1))
    self.n = np.arange(self.n[0]*U,self.n[-1]*U+1)
    tempX[::U] = self.x
    self.x = tempX
  @staticmethod
  def matchSignals(sig1,sig2):
    # combining and finding the min and max of both n arrays
    minIndex = np.min(np.append(sig1.n,sig2.n))
    maxIndex = np.max(np.append(sig1.n,sig2.n))
    # creating new n array with combined min and max
    newN = np.arange(minIndex, maxIndex+1, 1)
    # creating new Signal objects that are filled with 0s the size of the combined n array
    sig1filled = Signal(np.zeros(len(newN)), minIndex, maxIndex) 
    sig2filled = Signal(np.zeros(len(newN)), minIndex, maxIndex)
    # for each of the old n indexes it goes and fills in that value in the new array by shifting the index over by the combined n array's starting indexes difference with 0
    counter = 0
    shift = 0 - minIndex
    for i in sig1.n: 
      sig1filled.x[i+shift] = sig1.x[counter]
      counter += 1
    counter = 0 
    for i in sig2.n: 
      sig2filled.x[i+shift] = sig2.x[counter]  
      counter += 1
    return sig1filled, sig2filled


x = Signal([-1,2,1,-2],-3,0)
y = Signal([1,2,1,0,1],-2,2)

print('flip')
print(x.x)
x.plot()
x.flip()
print(x.x)
x.plot()

print('shift 1')
print(x.x)
x.plot()
x.shift(1)
print(x.x)
x.plot()

print('decimate 2')
print(x.x)
x.plot()
x.decimate(2)
print(x.x)
x.plot()

print('expand 3')
print(x.x)
x.plot()
x.expand(3)
print(x.x)
x.plot()