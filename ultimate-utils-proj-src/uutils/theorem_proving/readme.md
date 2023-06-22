# Ultimate Utils for Theorem proving 

This is where the common code for Brando's ML for Theorem Proving (ML4TP) will go. 

## Ideas

Ideas of what could go here:
- Tp Eval - Goal: given an LLM and a test/eval set, how good is it doing?
	- always start simple and make sure code runs/entire pipeline runs
	- LLM, try something simple and cheap to test code. Then we can try something more serious later.
		- idk if openai has something free with their api...
		- like taking in a HF model is good
	- metrics
		- ppl perplexity
		- string match - zero shot prove predict proof script tactic/isar/proof term, just string
			- average str match/token edit distance
			- exact string match
		- proving accuracy 
			- Prover 0 - zero shot prove predict proof script tactic/isar/proof term but using ITP env e.g. pisa
			- Prover1 =  LLM + Thor, Draft Sketch Prove (DSP), baldur, diversity model, etc.
		- (Prover + LLM (our own)) [later]
		- MiniF2F (Isabella, PISA) + prover + model (likely best/small debug)
	- ideally being able to include this in helm: https://crfm.stanford.edu/helm/latest/ 
- Data extraction tools?
