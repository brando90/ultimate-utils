#%%
"""
super: https://stackoverflow.com/questions/222877/what-does-super-do-in-python-difference-between-super-init-and-expl, https://realpython.com/python-super/#an-overview-of-pythons-super-function

subclass = if Child is a subclass of SomeBaseClass then the code from SomeBaseClass will be given to Child. e.g. a Square (more specific) is a subclass of a Rectangle (usually more general). https://www.codesdope.com/course/python-subclass-of-a-class/

super - basics with single inheritance: https://realpython.com/python-super/#a-super-deep-dive
- super(ChildClassName, self[instance of ChildClass]) can take two inputs the first the subclass & the second a specific instance of the subclass.
    - By including an instantiated object, super() returns a bound method: a method that is bound to the object, which gives the method the object’s context such as any instance attributes. 

In Python 3, the super(Square, self) call is equivalent to the parameterless super() call.



Goal: understand super(_BatchNorm, self).__init__(...)
"""

class Square:
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side**2

class Cube(Square):

    def surface_area(self) -> float:
        # same as super(Square, self) in python3
        area_one_face: float = super().area()
        return area_one_face * 6

    def volume(self):
        # face_area = super(Square, self).area()
        face_area = super().area()
        return face_area * self.length

    def _update_side(self, side: int) -> None:
        # super(Cube, self).__init__(side=side)  # for python 2
        super().__init__(side=side)
        assert(self.side == side), f'Should have updated the side to {side} but it\s: {self.side}'

c1: Cube = Cube(3)
print(c1.side)
c1._update_side(4)
print(c1.side)


#%%

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)