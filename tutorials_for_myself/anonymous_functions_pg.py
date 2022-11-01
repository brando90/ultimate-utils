# %%
"""
trying to detect which field is the anonymous function giving me isse since:
    AttributeError: Can't pickle local object 'FullOmniglot.__init__.<locals>.<lambda>'
doesn't tell me which one for some reason.
"""
import re
from pprint import pprint
from typing import Any, Callable, Union, Optional


def _is_anonymous_function(f) -> bool:
    """
    Returns true if it's an anonynouys function.

    ref: https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
    """
    return callable(f) and f.__name__ == "<lambda>"


def _get_anonymous_function_attributes(anything, halt: bool = False, verbose: bool = False) -> dict:
    """
    Returns the dictionary of name of fields to anonymous functions in the past anything thing.

    :param anything:
    :param halt:
    :param verbose:
    :return:
    """
    anons: dict = {}
    for field_name in dir(anything):
        field = getattr(anything, field_name)
        if _is_anonymous_function(field):
            if verbose:
                print(f'{field_name=}')
                print(f'{field=}')
            if halt:
                from pdb import set_trace as st
                st()
            anons[str(field_name)] = field
    return anons

def _get_anonymous_function_attributes_recursive(anything: Any, path: str = '') -> dict[str, Callable]:
    """"""
    anons: dict = {}
    def __get_anonymous_function_attributes_recursive(anything: Any,
                                                      path: Optional[str] = '',
                                                      ) -> None:
        if _is_anonymous_function(anything):
            # assert field is anything, f'Err not save thing/obj: \n{field=}\n{anything=}'
            # key: str = str(dict(obj=anything, field_name=field_name))
            key: str = str(path)
            anons[key] = anything
        else:
            for field_name in dir(anything):
                # if field_name != '__abstractmethods__':
                if not bool(re.search(r'__(.+)__', field_name)):
                    field = getattr(anything, field_name)
                    # only recurse if new field is not itself
                    if field is not anything:  # avoids infinite recursions
                        path_for_this_field = f'{path}.{field_name}'
                        __get_anonymous_function_attributes_recursive(field, path_for_this_field)
        return
    __get_anonymous_function_attributes_recursive(anything, path)
    return anons

class MyObj:
    def __init__(self):
        self.data = 'hi'
        self.anon = lambda x: x
        local_variable_me = 'my a local variable!'

    def non_anon(self, x):
        return x

class MyObj2:
    def __init__(self):
        self.data = 'hi'
        self.anon = lambda x: x
        local_variable_me = 'my a local variable!'

        self.obj = MyObj()

    def non_anon(self, x):
        return x


# obj = MyObj()
# print(f"{_get_anonymous_function_attributes(obj)=}")
# print()

top_obj = MyObj2()
# print(f'anons recursive: {_get_anonymous_function_attributes_recursive(obj)=}')
print('getting all anonymous functions recursively: ')
anons: dict = _get_anonymous_function_attributes_recursive(top_obj, 'top_obj')
print(f'{len(anons.keys())=}')
for k, v in anons.items():
    print()
    print(f'{k=}')
    print(f'{v=}')
    # print(k, v)
print()

from uutils import get_anonymous_function_attributes_recursive
get_anonymous_function_attributes_recursive(top_obj, 'top_obj', print_output=True)
print()
"""
Trying to fix: AttributeError: Can't pickle local object 'FullOmniglot.__init__.<locals>.<lambda>'
Trying to approximate with my obj and get: obj.__init__.<locals> to to get the obj.__ini__.<locals>.<lambda> 
"""
# since .<locals> seems like an attribute lets get it with dir (dir does "With an argument, attempt to return a list of valid attributes for that object.")
print(f'{"anon" in dir(obj)}')  # sanity check obj has anon
print(f'{dir(obj.__init__)=}')
print(f'{"anon" in dir(obj.__init__)}')  # failed
print(f'{type(obj.__init__.__self__)=}')
# weird...why is it such a odd type: type(obj.__init__.__self__)=<class '__main__.MyObj'>
# print(f'{dir(obj.__init__.__self__)=}')
print(f'{"anon" in dir(obj.__init__.__self__)}')
print(f'{_get_anonymous_function_attributes(obj.__init__.__self__)}')


# pickle it
