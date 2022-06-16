"""
ref:
"""
from collections import namedtuple

Point = namedtuple("Point", "x y")

point: Point = Point(x=2, y=3)
point_: Point = Point(x=2, y=3)
assert point == point_
print(f'{point=}')
print(f"{(point == point_)=}")


