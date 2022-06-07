import numpy as np
import matplotlib.pyplot as plt

def shift(self, nx=0):
  # y[n] = x[n-nx]
  self.iv = np.arange(self.iv[0] + nx, self.iv[-1] + nx + 1)

def decimate(self, D=2):
  # y[n] = x[Dn]
  self.dv = np.compress(self.iv % D == 0, self.dv)
  self.iv = np.compress(self.iv % D == 0,self.iv) // D

# Operator Overloading
def __add__(self, other):
  A, B = Signal(self, other)
  return Signal(A.dv + B.dv, A.iv[0], B.iv[-1])