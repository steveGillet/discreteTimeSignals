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
  def shift(self, n0 = 0):
    self.iv = np.arange(self.iv[0] + n0, self.iv[-1] + n0 + 1)
    return self
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
    return self
  
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
  def convolve(A, B):
    A, B = Signal.matchSignals(A, B)
    y = Signal(np.zeros(A.dv.size), A.iv[0], B.iv[-1])
    for x, k in zip(A.dv, A.iv):
      B_temp = Signal(B.dv, B.iv[0], B.iv[-1])
      B_temp.scale(x)
      B_temp.shift(k)
      y = y + B_temp
    return y

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
    self.dv = 1.0 * (self.iv == 0)

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



x = Sinusoid(0,10,omega=(8*np.pi / 9))
h = Impulse(0,50) - Impulse(0,50).scale(0.5).shift(1)
print(x)
print(h)
x.plot()
y = Signal.convolve(x, h)
y.plot()

omega = np.linspace(0, np.pi, 1000)
Hmag_z = np.sqrt(2 + 2 * np.cos(2 * omega))
Hmag_p = 1 / np.sqrt(1.6561 + 1.62 * np.cos(2 * omega))
iceCream = 0.905
Hmag = iceCream * Hmag_z * Hmag_p
plt.plot(omega, Hmag_p)
plt.plot(omega, Hmag_z)
plt.plot(omega, Hmag)
plt.show()

n = np.arange(0,50)
x = (1 + np.cos(3 * n * np.pi / 4))
h = np.array([1, 0, 1])
y = np.convolve(x, h)

plt.stem(y)
plt.show()

# x = Signal([-1,2,1,-2],-3,0)
# y = Signal([1,2,1,0,1],-2,2)

# p = Pulse(-3,3,2)
# p.plot()

# s = Sinusoid(-5,5,-9*np.pi/4)
# s.plot()

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