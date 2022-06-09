#%%
import re
s = 'Part 1. Part 2. Part 3 then more text'
x = re.search(r'Part 1\.(.*?)Part 3', s)
print(f'{x=}, {type(x)=}')
xx = re.search(r'Part 1\.(.*?)Part 3', s).group(1)
print(f'{xx=}')

ht = re.search(pattern=r'(fun n : nat => (.*?) : 0 + n = n)', string='(fun n : nat => eq_refl : 0 + n = n)')
print(f'{ht=}')

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

def get_single_ht(ppt: str, ept: str) -> str:
    # - put a re pattern that matches anything in place of the meta-variable ?GOAL is
    # pattern = r'\b?(\w)+\b'
    # pattern = r'?(\w)+'
    pattern_meta_var = r'\?(\w)+'
    # pattern = re.compile(r'\b(\w+)\s+\1\b')
    # repl = pattern.replace('\\', '\\\\')
    _ppt = re.sub(pattern=pattern_meta_var, repl='HERE', string=ppt)
    #
    pattern_any_proof_term = r'(.+)'
    _ppt = _ppt.replace('HERE', pattern_any_proof_term)
    # _ppt = "(fun n : nat => ?Goal : 0 + n = n)"
    # - now that the re pattern is in the place of the meta-var, compute the diff btw terms to get the ht that goes in the hole
    ht = re.search(_ppt, ept)
    print(f'{ept=}')
    pattern = r'(fun n : nat => (\w) : 0 + n = n)'
    pattern = r'(fun n : nat => (.*?) : 0 + n = n)'
    print(f'{pattern=}')
    ht = re.search(pattern=pattern, string=ept)
    # ht = re.search('(fun n : nat => (.*?) : 0 + n = n)', ept)
    ht = re.search(pattern=r'(fun n : nat => (.*?) : 0 + n = n)', string='(fun n : nat => eq_refl : 0 + n = n)')
    print(f'{ht=}')
    return ht

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