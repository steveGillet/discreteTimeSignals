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
    plt.xticks(x.n)
    plt.xlabel("n")
    plt.ylabel("values")
    plt.show()
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

x,y = Signal.matchSignals(x,y)
print(x.x)
print(x.n)
x.plot()
print(y.x)
print(y.n)
y.plot()