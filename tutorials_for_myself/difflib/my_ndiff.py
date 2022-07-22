# %%
import difflib

cases = [('afrykanerskojęzyczny', 'afrykanerskojęzycznym'),
         ('afrykanerskojęzyczni', 'nieafrykanerskojęzyczni'),
         ('afrykanerskojęzycznym', 'afrykanerskojęzyczny'),
         ('nieafrykanerskojęzyczni', 'afrykanerskojęzyczni'),
         ('nieafrynerskojęzyczni', 'afrykanerskojzyczni'),
         ('abcdefg', 'xac')]

for a, b in cases:
    print('{} => {}'.format(a, b))
    for i, s in enumerate(difflib.ndiff(a, b)):
        if s[0] == ' ':
            continue
        elif s[0] == '-':
            print(u'Delete "{}" from position {}'.format(s[-1], i))
        elif s[0] == '+':
            print(u'Add "{}" to position {}'.format(s[-1], i))
    print()

# %%
import difflib

# ppt = '(fun A B : Prop => ?Goal)'
ptp = 'Theorem th1: forall A B: Prop, A -> A -> (A->B) -> B.\n Proof.\n   intros A B.\n   intros a0 a1.\n   intros ab.\n   apply ab.\n   exact a1.\n'

ppt = '(fun A B : Prop => ?Goal)'
# ppt = '(fun A B : Prop => )'
re_ppt = '\\(fun\\ A\\ B\\ :\\ Prop\\ =>\\s*(.+)\\s*\\)'
ept = '(fun (A B : Prop) (_ a1 : A) (ab : A -> B) => ab a1)'

print(f'{ppt} => {ept}')
diff: list = list(difflib.ndiff(ppt, ept))
for i, s in enumerate(diff):
    if s[0] == ' ':
        continue
    elif s[0] == '-':
        print(u'Delete "{}" from position {}'.format(s[-1], i))
    elif s[0] == '+':
        print(u'Add "{}" to position {}'.format(s[-1], i))

print(''.join(difflib.restore(diff, 1)))
print(''.join(difflib.restore(diff, 2)))
print()

"""
algorithm so far:
- if its a lonely paren add it. i.e. paren by itself "(" or with only a space on either side "( " or " (" or " ( ", but then it stops being consecutive
    - this is a state looking for paren. 
    - todo: what if the diff is only a space? ignore it.
- if we are the lonely paren state but then the diff is greater than 3 we know it's not a lonely paren and move to the found a hole/diff state. 
still once we don't have a consecutive jump we are found the hole.
- for any char that is consecutive greater than size 2 (or 3?) continue until there is a non consecutive jump.

(- if not a lonely, then that is a hole no matter the size. it stops once the jump is none consecutive)
"""

