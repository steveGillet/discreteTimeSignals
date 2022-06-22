import numpy as np
import matplotlib.pyplot as plt

omega = np.linspace(0,np.pi,1000)
Hmag = np.sqrt(1.25 - np.cos(omega))
plt.plot(omega, Hmag)
plt.show()

x = Sinusoid(0,10,omega=(8*np.pi / 9))
x.plot()