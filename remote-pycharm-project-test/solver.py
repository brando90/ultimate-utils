import math
#==============this code added==================================================================:
# import sys
# sys.path.append("<PyCharm directory>/debug-egg/pydevd-pycharm.egg")
import pydevd_pycharm
pydevd_pycharm.settrace('9.59.196.91', port=12345, stdoutToServer=True, stderrToServer=True)
#================================================================================================
class Solver:

    def demo(self, a, b, c):
        d = b ** 2 - 4 * a * c
        if d > 0:
            disc = math.sqrt(d)
            root1 = (-b + disc) / (2 * a)
            root2 = (-b - disc) / (2 * a)
            return root1, root2
        elif d == 0:
            return -b / (2 * a)
        else:
            return "This equation has no roots"

if __name__ == '__main__':
    solver = Solver()

while True:
    a = int(input("a: "))
    b = int(input("b: "))
    c = int(input("c: "))
    result = solver.demo(a, b, c)
    print(result)
