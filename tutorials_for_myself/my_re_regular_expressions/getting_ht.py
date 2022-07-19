"""
Very nice interface for doing regexes: https://regex101.com/

"""
# import re
# for test in [
#   "(fun n : nat => eq_refl : 0 + n = n)",
#   "(fun n : nat => eq_refl : 0+n = n)",
#   "(fun n     :      nat     =>     eq_refl:0+n=n)",
# ]:
#     print(re.search(r'\(fun\s+n\s*:\s*nat\s+=>\s+(\w+)\s*:\s*0\s*\+\s*=\s*n\s*\)', test))
# import re
# prefix = re.escape('(fun n : nat => ')
# suffix = re.escape(' : 0 + n = n)')
# tester = re.compile(f'{prefix}(.+){suffix}')
# for test in [
#   "(fun n : nat => eq_refl : 0 + n = n)",
# ]:
#     print(tester.search(test))

import re
from pprint import pprint

prefix = re.escape('(fun n : nat => ')
suffix = re.escape(' : 0 + n = n)')
for test in [
    "(fun n : nat => eq_refl : 0 + n = n)",
]:
    print(re.search(f'{prefix}(.+){suffix}', test))
    print(re.search(f'{prefix}(.+){suffix}', test).group(1))

import re

ppt = '(fun n : nat => ?Goal : 0 + n = n)'
match = re.search(r'^(.*)\?\w+(.*)$', ppt)
re_ppt = f'{re.escape(match.group(1))}(.+){re.escape(match.group(2))}'
print(re_ppt)
print(re.match(re_ppt, "(fun n : nat => eq_refl : 0 + n = n)").groups())

# %%
import re

s = 'Part 1. Part 2. Part 3 then more text'
x = re.search(r'Part 1\.(.*?)Part 3', s)
print(f'{x=}, {type(x)=}')
xx = re.search(r'Part 1\.(.*?)Part 3', s).group(1)
print(f'{xx=}')

# %%
"""
Theorem add_easy_0:
forall n:nat,
  0 + n = n.
Proof.
    Show Proof.
    intros.
    Show Proof.
    simpl.
    Show Proof.
    reflexivity.
    Show Proof.
Qed.

(*
(fun n : nat => ?Goal : 0 + n = n)

(fun n : nat => eq_refl : 0 + n = n)
*)

want variable
ppt = "(fun n : nat => ?Goal : 0 + n = n)"
ept = "(fun n : nat => eq_refl : 0 + n = n)"
ht: str = get_ht(ppt, ept)
assert ht == "eq_refl"

"""
import re


# s = 'Part 1. Part 2. Part 3 then more text'
# re.search(r'Part 1\.(.*?)Part 3', s).group(1)

# def make_term_to_regex_term(term: str) -> str:
#     # pattern = r'(fun n : nat => (.*) : 0 + n = n)'.replace('(', '\(')
#     # pattern = f'r{term}'
#     pattern = f'{term}'
#     pattern = pattern.replace('(', r'\(')
#     pattern = pattern.replace(')', r'\)')
#     pattern = pattern.replace('+', r'\+')
#     print(f'{pattern=}')
#     return pattern

def get_single_ht(ppt: str, ept: str) -> str:
    # ppt = re.escape(ppt)  # to make everything literal since we want to make the re version of ppt
    print(f'{ppt=}')
    assert ppt == '(fun n : nat => ?Goal : 0 + n = n)'
    # - put a re pattern that matches anything in place of the meta-variable ?GOAL is
    pattern_meta_var = r'\?(\w)+'
    # re_ppt = re.sub(pattern=pattern_meta_var, repl='(.+)', string=ppt)
    _ppt = re.sub(pattern=pattern_meta_var, repl='HERE', string=ppt)
    _ppt = re.escape(_ppt)
    re_ppt = _ppt.replace('HERE', '(.+)')
    ans = '\\(fun\\ n\\ :\\ nat\\ =>\\ (.+)\\ :\\ 0\\ \\+\\ n\\ =\\ n\\)'
    assert re_ppt == ans, f'Failed, got {re_ppt=}\n wanted: {ans}'

    # - now that the re pattern is in the place of the meta-var, compute the diff btw terms to get the ht that goes in the hole
    out = re.search(pattern=re_ppt, string=ept)
    if out is None:
        raise ValueError(f'Output of ht search was {out=}, re ppt was {re_ppt=} and ept was {ept=}.')
    ht: str = out.group(1)
    print(f'{ht=}')
    return ht


print()
ppt = "(fun n : nat => ?Goal : 0 + n = n)"
ept = "(fun n : nat => eq_refl : 0 + n = n)"
ht: str = get_single_ht(ppt, ept)
assert ht == "eq_refl"

# %%
"""
Theorem add_easy_induct_1:
forall n:nat,
  n + 0 = n.
Proof.
  Show Proof.
      intros.
  Show Proof.
  induction n as [| n' IH].
  Show Proof.
  - simpl.
    Show Proof.
    reflexivity.
    Show Proof.
  - simpl.
    Show Proof.
    rewrite -> IH.
    Show Proof.
    reflexivity.
    Show Proof.
Qed.

(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) ?Goal
   (fun (n' : nat) (IH : n' + 0 = n') => ?Goal0) n)
   
(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) (eq_refl : 0 + 0 = 0)
   (fun (n' : nat) (IH : n' + 0 = n') =>
	eq_ind_r (fun n0 : nat => S n0 = S n') eq_refl IH : S n' + 0 = S n') n)

For multiple holes, we can have the model predict the content of all the holes.
Or mask as "noise" one of the holes and have it predict the other.
Or have it predict the first one, then put the contents of its prediction there
and then predict the next hole.
Or combination of those (e.g. noise mask a bunch of holes and predict a subset).
"""

ppt = 'abc HERE abc'
ept = 'abc TERM abc'
re_ppt = ppt.replace('HERE', '(.+)')
print()
print(f'{re_ppt=}')
out = re.search(pattern=re_ppt, string=ept)
print(out)
print(out.groups())

ppt = 'abc HERE abc HERE abc'
ept = 'abc TERM1 abc TERM2 abc'
re_ppt = ppt.replace('HERE', '(.+)')
print()
print(f'{re_ppt=}')
out = re.search(pattern=re_ppt, string=ept)
print(out)
print(out.groups())

print()
ppt = """(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) ?Goal"""
print(f"{ppt=}")
ept = """(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) (eq_refl : 0 + 0 = 0)"""
print(f'{ept=}')
pattern_meta_var = r'\?(\w)+'
_ppt = re.sub(pattern=pattern_meta_var, repl='HERE', string=ppt)
print(f'{ppt=}')
_ppt = re.escape(_ppt)
print(f'{ppt=}')
re_ppt = _ppt.replace('HERE', '(.+)')
print(f'{re_ppt=}')
out = re.search(pattern=re_ppt, string=ept)
print(out)
print(out.groups())

print()

# sometimes the actual proof term missing won't have white spaces surrounding it but the ppt will have surrounding spaces where the hole
# would be. So in goal cames I removed the surrounding whitespaces. Then inserted a regex that accepts a hole with or
# without surrounding white spaces. That way in case the proof term in the hole does have surrounding white spaces then
# the regex hole catcher would match it anyway.
ppt = """\n   (fun (n' : nat) (IH : n' + 0 = n') => ?Goal0) n)"""
ept = """\n   (fun (n' : nat) (IH : n' + 0 = n') =>\n\teq_ind_r (fun n0 : nat => S n0 = S n') eq_refl IH : S n' + 0 = S n') n)"""
print(f"{ppt=}")
print(f'{ept=}')
pattern_meta_var = r'\s*\?(\w)+\s*'
_ppt = re.sub(pattern=pattern_meta_var, repl='HERE', string=ppt)
print(f'{_ppt=}')
_ppt = re.escape(_ppt)
print(f'{_ppt=}')
re_ppt = _ppt.replace('HERE', '\s*(.+)\s*')
print(f'{re_ppt=}')
out = re.search(pattern=re_ppt, string=ept)
print(out)
assert out is not None, f'expected two holes matched but go {out=}'
print(out.groups())

print()
ppt = """(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) ?Goal
   (fun (n' : nat) (IH : n' + 0 = n') => ?Goal0) n)"""
print(f"{ppt=}")
ept = """(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) (eq_refl : 0 + 0 = 0)
   (fun (n' : nat) (IH : n' + 0 = n') =>
	eq_ind_r (fun n0 : nat => S n0 = S n') eq_refl IH : S n' + 0 = S n') n)"""
print(f'{ept=}')
pattern_meta_var = r'\s*\?(\w)+\s*'
_ppt = re.sub(pattern=pattern_meta_var, repl='HERE', string=ppt)
print(f'{_ppt=}')
_ppt = re.escape(_ppt)
print(f'{_ppt=}')
re_ppt = _ppt.replace('HERE', '\s*(.+)\s*')
print(f'{re_ppt=}')
out = re.search(pattern=re_ppt, string=ept)
print(out)
print(out.groups())


def get_multiple_hts(ppt: str, ept: str) -> tuple[str]:
    # - put a re pattern that matches anything in place of the meta-variable ?GOAL is
    pattern_meta_var = r'\s*\?(\w)+\s*'
    _ppt = re.sub(pattern=pattern_meta_var, repl='HERE', string=ppt)
    print(f'{_ppt=}')
    _ppt = re.escape(_ppt)
    print(f'{_ppt=}')
    re_ppt = _ppt.replace('HERE', r'\s*(.+)\s*')
    print(f'{re_ppt=}')

    # - now that the re pattern is in the place of the meta-var, compute the diff btw terms to get the ht that goes in the hole
    print(f'{ept=}')
    out = re.search(pattern=re_ppt, string=ept)
    if out is None:
        raise ValueError(f'Output of ht search was {out=}, re ppt was {re_ppt=} and ept was {ept=}.')
    hts: tuple[str] = out.groups()
    return hts

print('\n\n --------------')
ppt = """(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) ?Goal
   (fun (n' : nat) (IH : n' + 0 = n') => ?Goal0) n)"""
ept = """(fun n : nat =>
 nat_ind (fun n0 : nat => n0 + 0 = n0) (eq_refl : 0 + 0 = 0)
   (fun (n' : nat) (IH : n' + 0 = n') =>
	eq_ind_r (fun n0 : nat => S n0 = S n') eq_refl IH : S n' + 0 = S n') n)"""
hts = get_multiple_hts(ppt, ept)
assert hts is not None, f'expected two holes matched but go {hts=}'
from pprint import pprint
pprint(hts)
print()

#%%
"""
compute "differences" of strings.
"""
import re

ppt = 'abc HERE abc'
ept = 'abc TERM abc'
re_ppt = ppt.replace('HERE', '(.+)')
print()
print(f'{re_ppt=}')
out = re.search(pattern=re_ppt, string=ept)
print(out)
print(out.groups())

ppt = 'abc HERE abc HERE abc'
ept = 'abc TERM1 abc TERM2 abc'
re_ppt = ppt.replace('HERE', '(.+)')
print()
print(f'{re_ppt=}')
out = re.search(pattern=re_ppt, string=ept)
print(out)
print(out.groups())