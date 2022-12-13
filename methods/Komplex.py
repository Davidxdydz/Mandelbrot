import math


class Komplex:
    def __init__(self, a, b):
        self.a = float(a) # Realteil
        self.b = float(b) # Imaginaerteil

    def real(self):
        return self.a

    def img(self):
        return self.b

    def abs(self):
        return math.sqrt(self.a**2 + self.b**2)

    def konjugiert(self):
        return Komplex(self.a, -self.b)

    def __add__(self, y):
        return Komplex(self.a + y.real(), self.b + y.img())

    def __sub__(self, y):
        return Komplex(self.a - y.real(), self.b - y.img())

    def __mul__(self, y):
        return Komplex(self.a*y.real() - self.b*y.img(), self.a*y.img() + self.b*y.real())

    def __truediv__(self, y):
        return Komplex((self.a*y.real() + self.b*y.img())/y.abs()**2, (self.b*y.real() - self.a*y.img())/y.abs()**2)

    def __str__(self):
        if self.b >= 0:
            return str(self.a) + " + i" + str(self.b)
        else:
            return str(self.a) + " - i" + str(abs(self.b))

    def __abs__(self):
        return self.abs()
