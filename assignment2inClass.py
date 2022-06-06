import numpy as np
import matplotlib.pyplot as plt

([])
"""a way to comment!?"""
class Signal:
  def __init__(self, x, start, end):
    self.x = np.array(x)
    self.n = np.arange(start,end+1,1)
  def plot(self,ylabel="x[n]"):
    plt.stem(self.n,self.x)
    plt.xticks(self.n)
    plt.xlabel("n")
    plt.ylabel(ylabel)
    plt.show()
  #directive
  @staticmethod
  def matchSignal(A,B):
    xA = A.x
    xB = B.x
    nA = A.n
    nB = B.n
    n = np.arange(np.min([nA[0], nB[0]]), np.max([nA[-1], nB[-1]]) + 1)
    # x = np.pad(x,(left,right)).
    xA = np.pad(xA, (0 if nA[0] <= n[0] else nA[0] - n[0], 0 if nA[-1] >= n[-1] else nA[-1] + n[-1]))
    xB = np.pad(xB, (0 if nB[0] <= n[0] else nB[0] - n[0], 0 if nB[-1] >= n[-1] else nB[-1] + n[-1]))
    return Signal(xA, n[0], n[-1]), Signal(xB, n[0], n[-1])
  def __add__(self,other):
    self,other = Signal.matchSignal(self,other)
    x = self.x + other.x
    n = self.n
    return(Signal(x,n[0],n[-1]))

x = Signal([-1,2,1,-2],-3,0)
x.plot()
y = Signal([0,1,2,3],-2,1)
x,y=Signal.matchSignal(x,y)
print(x)
x.plot()