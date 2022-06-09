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
prefix = re.escape('(fun n : nat => ')
suffix = re.escape(' : 0 + n = n)')
for test in [
  "(fun n : nat => eq_refl : 0 + n = n)",
]:
    print(re.search(f'{prefix}(.+){suffix}', test))
    print(re.search(f'{prefix}(.+){suffix}', test).group(1))

#%%
import re
s = 'Part 1. Part 2. Part 3 then more text'
x = re.search(r'Part 1\.(.*?)Part 3', s)
print(f'{x=}, {type(x)=}')
xx = re.search(r'Part 1\.(.*?)Part 3', s).group(1)
print(f'{xx=}')

#%%
"""
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
#s = 'Part 1. Part 2. Part 3 then more text'
#re.search(r'Part 1\.(.*?)Part 3', s).group(1)

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


#%%
"""
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