import numpy as np
import matplotlib.pyplot as plt
from DiscreteTime import Signal

class Impulse(Signal):
  def __init__(self,start,end):
    self.model = lambda n : 1 * (n == 0)
    self.iv = np.arange(start, end+1)
    self.dv = self.model(self.iv)

class Step(Signal):
  def __init__(self,start,end):
    self.model = lambda n : 1 * (n >= 0)
    self.iv = np.arange(start, end+1)
    self.dv = self.model(self.iv)

class Pulse(Signal):
  def __init__(self,start,end, N=1):
    self.model = lambda n : 1 * np.logical_and(n >= 0, n < N)
    self.iv = np.arange(start, end+1)
    self.dv = self.model(self.iv)

class Sinusoid(Signal):
  def __init__(self, start, end, A=1, omega=(np.pi/9), theta=0):
    self.model = lambda n : A * np.cos(omega * n + theta)
    self.iv = np.arange(start, end + 1)
    self.dv = self.model(self.iv)

class ComplexExp(Signal):
  def __init__(self, start, end, A=1, omega=(np.pi/9), theta=0):
    self.model = lambda n : A * np.exp(1j * (omega * n + theta))
    self.iv = np.arange(start, end + 1)
    self.dv = self.model(self.iv)
  def plot(self,ylab="x[n]",title="Complex Exponential", real=True):
    plt.stem(self.iv,np.real(self.dv) if real else np.imag(self.dv))
    plt.show()

class System:
  def __init__(self,T):
    self.T = T

  def evaluate(self, x):
    return Signal(self.T(x), x.iv[0], x.iv[-1])

x = Signal([-2, 1, 0, -1], -2, 1)
# offset
sys1 = System(lambda x : x.dv + 1)
#squarer
sys2 = System(lambda x : x.dv ** 2)
#shifter
sys3 = System(lambda x : x.shift(2).dv)
y = sys3.evaluate(x)
x.plot()
y.plot()

# c = Sinusoid(-10,10)
# c.plot()

# c1 = ComplexExp(-10,10)
# c1.plot()

# p = Pulse(-5,5,3)
# p.plot()