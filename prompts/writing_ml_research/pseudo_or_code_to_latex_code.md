# ---- Write my pseudocode into latex code

## Prompt: Pseudocode to latex code
Write a top quality latex pseudocode section for a NeurIPS NIPS ICML ICLR AAAI CVPR machine learning publication 
from my pseudocode in pseudo python:, 
```python
#my pseudo python
```
here is an example of the target pseudocode in latex:
```latex
% example pseudo code in latex
\begin{algorithm} % \label{greedy_synthesis_bfs}
	\caption{Greedy Synthesis of Types in BFS order} 
	\begin{algorithmic}[1]
	    \State{\textbf{Input: } batch of terms $B$ to synthesize types of length $|B|$}
	    \State $\tau = [] $ initialize the empty list to hold the predicted batch of types.
		\For {$b = 1,2,...,|B|$}
		    \State $T_b = $ get a term from a batch of terms $B$ with index $b$
		    \State $R_b = $ get the predicted rule sequence $R_b$ for the term $T_b$
		    \State $G_b = [] $ initialize an empty list to hold the greedy rules from $R_b$ 
			\For {$t = 1,2, ..., |R_b|$}
			    \State $p_{b,i}$ = get distribution of possible rules for the current type inference step $t$
			    \State $\hat r_{b, t} = arg \max_{r \in {1,2,...,|p_{b,i}|} } \{ p_{b,t}[r] \}$ i.e., get the most likely rule for this time step $t$ for the current term $T_b$
			    \State $G_b[t] = r_{b, t}$ i.e., store the greedy rule to predict in time step $t$
			\EndFor
			\State $\tau_b = $ generate type by applying the greedy rules from $G_b$ in BFS order. 
			If a rule cannot be applied with respect to the previous rule in BFS order skip this type and append the dummy error type.
			\State $\tau[b] = \tau_b$ append the predicted type $\tau_b$ to the list of types.
		\EndFor
		\State{\textbf{Return: } batch of predicted types for input terms $\tau$}
	\end{algorithmic} 
\end{algorithm}
```
The output of the latex pseudocode or algorithms is for top quality publication in NeurIPS NIPS ICML ICLR AAAI CVPR machine learning.
Do not change citations or urls e.g. \citep{...}, urls.
Do not change the variable names but do make them compatible with latex.
Do not change any part that is already excellent.
Provide 3 latex pseudo codes / algorithms from my pseudo python:

# -- notes --
Do not sound exaggerated or pompous.
Do not change the facts in my example.
Keep it concise, scientific, direct, use the active voice.
Follow the instructions conditioned by the example provided above.
The discussion section should be similar in length to the top example i.e. 1, 2 or 3 paragraphs.
Provide 3 re-phrased options: