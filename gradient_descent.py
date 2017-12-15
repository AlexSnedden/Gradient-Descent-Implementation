import math
import numpy as np

def partial_deriv(f, x, idx):
    h = 0.000001
    x_ = np.array([v + h if i == idx else v for i,v in enumerate(x)])
    return (f(x_) - f(x))/h

class gradientDescentOp:
    def __init__(self, f, x, lr, m):
        self.f = f
        self.x = x
        self.lr = lr
        self.m = m
        self.current_min = self.f(self.x)
        self.prev_updates = np.zeros(len(x))
    def update(self):
        x = list(self.x)
        for i in xrange(len(self.x)):
            gradient_update = -self.lr*partial_deriv(self.f, x, i)
            self.x[i] += self.m*self.prev_updates[i]
            self.x[i] +=  gradient_update
            self.prev_updates[i] = gradient_update
        self.current_min = self.f(self.x)
    def current(self):
        return self.current_min
