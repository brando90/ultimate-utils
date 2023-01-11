#eval Lean.versionString

-- W = WFF
-- Wn = NNF, not appears before a prop symbol

inductive WFF (α : Type) : Type
| atom: α → WFF α
| not: WFF α → WFF α
| and: WFF α → WFF α → WFF α
| or: WFF α → WFF α → WFF α
| imp: WFF α → WFF α → WFF α
| iff: WFF α → WFF α → WFF α

inductive NNF (α : Type) : Type
| atom: α → NNF α
| not: α → NNF α
| and: NNF α → NNF α → NNF α
| or: NNF α → NNF α → NNF α
| imp: NNF α → NNF α → NNF α
| iff: NNF α → NNF α → NNF α

def WFF.toNNF {α : Type} : WFF α → NNF α := sorry

def NNF.toWFF {α : Type} : NNF α → WFF α
| (NNF.atom a) => WFF.atom a
| (NNF.not a) => WFF.not (WFF.atom a)
| (NNF.and a b) => WFF.and (NNF.toWFF a) (NNF.toWFF b)
| (NNF.or a b) => WFF.or (NNF.toWFF a) (NNF.toWFF b)
| (NNF.imp a b) => WFF.imp (NNF.toWFF a) (NNF.toWFF b)
| (NNF.iff a b) => WFF.iff (NNF.toWFF a) (NNF.toWFF b)

-- forall x1: f.Type, f(x1) = f(x2) implies x1 = x2
-- goal: show NNF <= WFF
theorem NNF_injective_to_WFF (x y : NNF α):
  -- NNF.toWFF x = NNF.toWFF y → x = y := 
  x.toWFF = y.toWFF → x = y := 
  by induction x generalizing y
     . simp [NNF.toWFF] 
       sorry


