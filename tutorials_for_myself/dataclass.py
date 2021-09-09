# https://realpython.com/python-data-classes/

from dataclasses import dataclass

@dataclass
class DataClassCard:
    rank: str
    suit: str
    default_test: str = 'default'

card = DataClassCard('Q', 'Hearts')

print(card)

print(card == DataClassCard('Q', 'Hearts'))

#%%

from dataclasses import dataclass, field

def get_str():
    return 'default'

def get_lst():
    return [1, 2]

@dataclass
class DataClassCard:
    rank: str
    suit: str

    # - you need to be careful with mutable objects otherwise if you change it in one class it changes in all of them!
    # luckly python will throw an error if you try to do that...

    # default_test: str = field(default_factory=get_str)
    # default_test: str = [1, 2]
    default_test: str = field(default_factory=get_lst)


card = DataClassCard('Q', 'Hearts')
print(card)
print(card.default_test)