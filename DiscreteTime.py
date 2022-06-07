import numpy as np
from numpy import int64
import matplotlib.pyplot as plt

class Signal:
  def __init__(self, DVin, IVstart, IVend):
    self.dv = np.array(DVin)
    self.iv = np.arange(IVstart,IVend+1,1)
  def plot(self):
    plt.stem(self.iv, self.dv)
    plt.xticks(self.iv)
    plt.xlabel("n")
    plt.ylabel("values")
    plt.show()
  def flip(self):
    self.dv = np.flip(self.dv)
    self.iv = -np.flip(self.iv)
  def shift(self,n0):
    if n0 >= 0:
      self.dv[n0:] = self.dv[:-n0]
      self.dv[:n0] = 0
    else:
      self.dv[:n0] = self.dv[-n0:]
      self.dv[n0:] = 0
  def decimate(self, D):
    indShift = 0 - self.iv[0]
    tempdv = np.compress((self.iv % D) == 0,self.dv)
    tempiv = np.compress((self.iv % D) == 0,self.iv)
    self.dv = np.zeros((len(self.dv)))
    for i in range(len(tempiv)):
      for j in range(len(self.dv)):
        if(j==tempiv[i]+indShift):
          self.dv[j] = tempdv[i]
  def expand(self,U):
    tempdv = np.zeros(U*len(self.dv)-(U-1))
    self.iv = np.arange(self.iv[0]*U,self.iv[-1]*U+1)
    tempdv[::U] = self.dv
    self.dv = tempdv
  def scale(self, k):
    self.dv *= k
  
  # Operator Overloading
  def __add__(self, other):
    A, B = Signal.matchSignals(self, other)
    return Signal(A.dv + B.dv, A.iv[0], B.iv[-1])
  
  def __sub__(self, other):
    A, B = Signal.matchSignals(self, other)
    return Signal(A.dv - B.dv, A.iv[0], B.iv[-1])

  def __mul__(self, other):
    A, B = Signal.matchSignals(self, other)
    return Signal(A.dv * B.dv, A.iv[0], B.iv[-1])
  
  def __eq__(self, other):
    A, B = Signal.matchSignals(self, other)
    return np.array_equal(A.dv,B.dv)

  @staticmethod
  def matchSignals(sig1,sig2):
    # combining and finding the min and max of both n arrays
    minIndex = np.min(np.append(sig1.iv,sig2.iv))
    maxIndex = np.max(np.append(sig1.iv,sig2.iv))
    # creating new n array with combined min and max
    newiv = np.arange(minIndex, maxIndex+1, 1)
    # creating new Signal objects that are filled with 0s the size of the combined n array
    sig1filled = Signal(np.zeros(len(newiv)), minIndex, maxIndex) 
    sig2filled = Signal(np.zeros(len(newiv)), minIndex, maxIndex)
    # for each of the old n indexes it goes and fills in that value in the new array by shifting the index over by the combined n array's starting indexes difference with 0
    counter = 0
    shift = 0 - minIndex
    for i in sig1.iv: 
      sig1filled.dv[i+shift] = sig1.dv[counter]
      counter += 1
    counter = 0 
    for i in sig2.iv: 
      sig2filled.dv[i+shift] = sig2.dv[counter]  
      counter += 1
    return sig1filled, sig2filled

class Impulse(Signal):
  def __init__(self, IVstart, IVend):
    self.iv = np.arange(IVstart, IVend+1)
    self.dv = 1 * (self.iv == 0)

class Step(Signal):
  def __init__(self, IVstart, IVend):
    self.iv = np.arange(IVstart, IVend+1)
    # 1* makes the boolean into int
    self.dv = 1 * (self.iv >= 0)

class Pulse(Signal):
  def __init__(self, IVstart, IVend, N):
    self.iv = np.arange(IVstart, IVend+1)
    # 1* makes the boolean into int
    self.dv = np.zeros(len(self.iv))
    for i in range (len(self.iv)):
      self.dv[i] = 1 * (self.iv[i] >= 0 and self.iv[i] <= N)

class PowerLaw(Signal):
  def __init__(self, IVstart, IVend, A = 1, alpha = 1):
    self.iv = np.arange(IVstart, IVend+1)
    # 1* makes the boolean into int
    self.dv = A * alpha ** self.iv 

class Sinusoid(Signal):
  def __init__(self, IVstart, IVend, omega = 1):
    self.iv = np.arange(IVstart, IVend+1)
    # 1* makes the boolean into int
    self.dv = np.cos(omega*self.iv) 


x = Signal([-1,2,1,-2],-3,0)
y = Signal([1,2,1,0,1],-2,2)

p = Pulse(-3,3,2)
p.plot()

s = Sinusoid(-5,5,-9*np.pi/4)
s.plot()

# d = Impulse(-3,3)
# d.scale(3)
# d.plot()

# w = x * d
# w.plot()

# s = Step(-3, 3)
# s.shift(-2)
# s.plot()

# w2 = x * s
# w2.plot()

# pl = PowerLaw(A=5, alpha = -0.75, IVstart = 0, IVend=5)
# pl.plot()

# print('flip')
# print(x.x)
# x.plot()
# x.flip()
# print(x.x)
# x.plot()

# print('shift 1')
# print(x.x)
# x.plot()
# x.shift(1)
# print(x.x)
# x.plot()

# print('decimate 2')
# print(x.x)
# x.plot()
# x.decimate(2)
# print(x.x)
# x.plot()

# print('expand 3')
# print(x.x)
# x.plot()
# x.expand(3)
# print(x.x)
# x.plot()