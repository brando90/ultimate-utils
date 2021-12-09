print('running module __init__.py from folder pkg1')

from . import imported_module
from .imported_module_by_front_dot import purpose

CONST1 = 1 # can be accessed as pkg1.CONST1 if you import pkg1 with import pkg1