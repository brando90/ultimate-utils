# %%
"""
trying to detect which field is the anonymous function giving me isse since:
    AttributeError: Can't pickle local object 'FullOmniglot.__init__.<locals>.<lambda>'
doesn't tell me which one for some reason.
"""
from pprint import pprint


def _is_anonymous_function(f) -> bool:
    """
    Returns true if it's an anonynouys function.

    ref: https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
    """
    return callable(f) and f.__name__ == "<lambda>"


class MyObj:
    def __init__(self):
        self.data = 'hi'
        self.anon = lambda x: x

    def non_anon(self, x):
        return x


obj = MyObj()
print(f'{dir(obj)=}')
print(f'{vars(obj)=}')
print(f'{obj.__dict__=}')
print(f'{obj.__dir__=}')
print(f'{obj.__dir__()=}')
print(f'{bool("anon" in dir(obj))}')
print(f'{bool("data" in dir(obj))}')
# print(f'{locals(obj)=}')

pprint(vars(obj))
# [print(getattr(obj, field)) for field in dir(obj)]
for field_name in dir(obj):
    print(f'{field_name=}')
    field = getattr(obj, field_name)
    if _is_anonymous_function(field):
        print(field)

# pickle it
