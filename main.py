class T(object):
    def __init__(self):
        print('T init')

class TT(object):
    def x(self):
        print('TT x')

class T1(T, TT):
    def __init__(self):
        super(T1, self).__init__()
        print('T1 init')

    def x(self):
        super(T1, self).x()
        print('T1 x')

t1 = T1()
t1.x()
