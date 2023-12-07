import numpy as np
import math

class Potential(object):
    def __init__(self, xx, yy):
        self.xx, self.yy = xx, yy
    def gaussian(self, a, b, coeff):
        V = ((self.xx - a) ** 2 + (self.yy - b) ** 2) / 2.0 * coeff
        return V

    def double_well(self, a1,b1,a2,b2,c, coeff):
        V = np.exp(-c * ((self.xx - a1) ** 2 + (self.yy - b1) ** 2)) + np.exp(-c * ((self.xx - a2) ** 2 + (self.yy - b2) ** 2))
        V = 1 / (V + (self.xx - 0.5) ** 2 + (self.yy - 0.5) ** 2 + 1)
        V = -np.log(V)
        return V*coeff

    def trig(self, a, b, coeff):
        V = 1 - np.sin(self.xx*a*np.pi)*np.sin(self.yy*b*np.pi)
        return V*coeff/2
