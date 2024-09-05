"""
For prompt generation, it's ok to use {placeholder} in the prompt string but don't use .format(placeholder=x), instead split it by {placeholder} and  then concatenate the parts. 
The reason is if the string is for maths & has latex, .format() will get confused because there will be curly braces for meant for string replacement. 

Processing the answer is prompt depedent, so we are pairing the process answer code for each prompt here. 
"""
from typing import Union

# -- Prompt Minerva MATH
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L22
H_MATH_MINERVA_PROMPT_TEMPLATE = (
r"""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct."""
)

def get_prob_str_minerva_prompt(data_pt: dict, prompt_template: str = H_MATH_MINERVA_PROMPT_TEMPLATE) -> str:
    # note: we are using string concatenation to avoid python getting confused with curly brackets for latex vs python string placeholders
    return prompt_template + "\n\n" + "Problem:" + "\n" + data_pt['problem'] + "\n\n" + "Solution:"

def extract_answer_minerva_prompt(completion: str) -> Union[None, str]:
    """ Extracts the boxed answer or None (if no boxed answer), removes box and cleans leading/training white spaces. e.g., 
            $$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$ -> "-\frac{2}{3}"
            $$hello world$$ -> None  # no boxed answer
    """
    from evals.utils import last_boxed_only_string, remove_boxed
    boxed_answer: str = last_boxed_only_string(completion)
    extracted_answer: str = remove_boxed(boxed_answer)
    extracted_answer = extracted_answer.strip() if extracted_answer is not None else None
    return extracted_answer

# -- Prompt Minerva MATH 2 - Better
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L22
H_MATH_MINERVA_PROMPT_TEMPLATE_2_BETTER = (
r"""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$"""
)

# -- Prompt minerva MATH 3 - better minerva + cot/scratch_pad
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L22

def get_prob_str_minerva_prompt_cot(data_pt: dict, prompt_template: str = H_MATH_MINERVA_PROMPT_TEMPLATE) -> Union[str, None]:
    # Note: instead of using .format(name=value) we are using the exact string for the placeholder so that the .format function doesn't get confused with curly braces.
    return prompt_template.replace("{problem}", data_pt['problem'])

def extract_answer_from_list_completion_strings_mv(completions_strs_per_prompt: list[str]) -> Union[str, None]:
    """ Extract a single answer str from a list of completions strings for a single prompt via majority voting (mv). """
    from evals.utils import last_boxed_only_string, remove_boxed
    model_answers: list[Union[str, None]] = []
    for completion_str_per_prompt in completions_strs_per_prompt:  # for each completion string for a single prompt
        if isinstance(completion_str_per_prompt, list):
            print(f'--> Warning: your are using majority voting (mv) and {len(completion_str_per_prompt)=} is 1 or less!')  if len(completion_str_per_prompt) <= 1 else None
        boxed_answer: Union[str, None] = last_boxed_only_string(completion_str_per_prompt)
        extracted_answer: Union[str, None] = remove_boxed(boxed_answer)
        extracted_answer = extracted_answer.strip() if extracted_answer is not None else None
        model_answers.append(extracted_answer)
    # Get the majority voted answer
    # if all model answers is none return None (eric's edge case)
    if all([ans is None for ans in model_answers]):
        return None
    from collections import Counter
    c = Counter(model_answers)  # for list of possible answers, produce count how many times each answer str appears
    majority_voted_answer: Union[str, None] = c.most_common(1)[0][0]  # 1 for most common, 0 since it returns a list of tuples (val, count), last 0 for the (most common) value
    return majority_voted_answer

STOP_TOKENS: list[str] = ["Solution:", "Problem:", "Question:", "USER:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
# STOP_TOKENS: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS: list[str] = ["Question:", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
# STOP_TOKENS_worse: list[str] = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]

# -- Prompt Meta MATH https://github.com/meta-math/MetaMath/blob/main/eval_math.py#L38 
# for reference but due to {instruction} placeholder might give problems.
PROBLEM_PROMPT_META_MATH = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)
def get_default_meta_math_prompt(h_math_dict: dict, prompt_template: str = PROBLEM_PROMPT_META_MATH) -> str:
    """"""
    # note: we are using string concatenation to avoid python getting confused with curly brackets for latex vs python string placeholders
    part1, part2 = prompt_template.split("{instruction}")
    # return prompt_template.format(instruction=h_math_dict["problem"])  # DONT USE
    return part1 + h_math_dict["problem"] + part2

def process_results_meta_math(doc: str, completion: str, answer: str) -> bool:
    """
    Process the results of the model's completion and compare it with the expected answer.

    Args:
        doc (str): The input document or question.
        completion (str): The model's completion or generated response.
        answer (str): The expected answer.

    Returns:
        bool: True if the extracted answer is equivalent to the expected answer, False otherwise.
    """
    # Split the completion string by 'The answer is: '
    split_ans = completion.split('The answer is: ')
    
    # Check if the completion string contains 'The answer is: '
    if len(split_ans) > 1:
        # Get the last part of the split completion string
        ans = split_ans[-1]
        
        # Split the answer string by '.\n' and take the first part
        extract_ans_temp = ans.split('.\n')[0]
        
        # Strip leading/trailing whitespace from the extracted answer
        extract_ans_temp = extract_ans_temp.strip()
        
        # Check if the extracted answer ends with a dot
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            # Remove the trailing dot from the extracted answer
            extract_ans = extract_ans_temp[0:-1]
        else:
            # Use the extracted answer as is
            extract_ans = extract_ans_temp
        
        # Strip leading/trailing whitespace from the final extracted answer
        extract_ans = extract_ans.strip()
        
        # Check if the extracted answer is equivalent to the expected answer
        if is_equiv(extract_ans, answer):
            # Return True if the answers are equivalent
            return True
        else:
            # Return False if the answers are not equivalent
            return False
    else:
        # Return False if 'The answer is: ' is not found in the completion string
        return False

# -- HELM prompt, 8 shot, CoT? ref: https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/runs/v1.0.0/math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model=01-ai_yi-34b/scenario_state.json, https://crfm.stanford.edu/helm/lite/latest/#/runs/math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,model=01-ai_yi-34b
# HELM_MATH_PROMPT_8SHOT_COT1_TEMPLATE: str = (
# """Given a mathematics problem, determine the answer. Simplify your answer as much as possible.###
# Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
# Answer: First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}.###
# Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction.
# Answer: Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}.###
# Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$?
# Answer: Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$###
# Problem: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?
# Answer: Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\\cdot800=19\\cdot200=\\boxed{3800}$ feet.###
# Problem: If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$?
# Answer: We know that $(x + y)^2 = (x^2 + y^2) + 2xy = 25$. We are given that $xy = 6$. So, by substitution, $x^2 + y^2 + 2xy = x^2 + y^2 + 2(6) = 25$. It follows that $x^2 + y^2 = 25 - 12 = \\boxed{13}$.###
# Problem: On a hot day, Megan likes to eat a Popsicle every 15 minutes. Assuming she keeps up that rate of consumption, how many Popsicles can Megan finish in 4 hours and 30 minutes?
# Answer: Let $p$ be the number of Popsicles Megan can finish in 4 hours and 30 minutes. If we convert that period of time into minutes, we find that 4 hours and 30 minutes is equal to $(4)(60)+30=270$ minutes. From here, we can set up the proportion \\begin{align*} \\frac{x}{270}& =\\frac{1}{15}\\\\\\Rightarrow \\qquad x& =\\left(\\frac{1}{15}\\right)(270)\\\\\\Rightarrow \\qquad x& =\\boxed{18}\\end{align*}###
# Problem: Compute $95^2$ in your head.
# Answer: We have $(90 + 5)^2 = 90^2 + 2(90)(5) + 5^2 = 8100 + 900 + 25 = \\boxed{9025}$.###
# Problem: If $2^8=16^x$, find $x$.
# Answer: We can write $16$ as $2^4$. Therefore, we can write our equation as $2^8 = 2^{4 \\cdot x}$. Solving, we get that $x = \\boxed{2}$.###
# Problem: {problem}
# Answer:""")

# HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE: str = (
# """Given a mathematics problem, determine the answer. Simplify your answer as much as possible.###
# Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
# Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}.###
# Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction.
# Solution: Let's think step by step. Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}.###
# Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$?
# Solution: Let's think step by step.Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$###
# Problem: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?
# Solution: Let's think step by step. Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\\cdot800=19\\cdot200=\\boxed{3800}$ feet.###
# Problem: If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$?
# Solution: Let's think step by step. We know that $(x + y)^2 = (x^2 + y^2) + 2xy = 25$. We are given that $xy = 6$. So, by substitution, $x^2 + y^2 + 2xy = x^2 + y^2 + 2(6) = 25$. It follows that $x^2 + y^2 = 25 - 12 = \\boxed{13}$.###
# Problem: On a hot day, Megan likes to eat a Popsicle every 15 minutes. Assuming she keeps up that rate of consumption, how many Popsicles can Megan finish in 4 hours and 30 minutes?
# Solution: Let's think step by step. Let $p$ be the number of Popsicles Megan can finish in 4 hours and 30 minutes. If we convert that period of time into minutes, we find that 4 hours and 30 minutes is equal to $(4)(60)+30=270$ minutes. From here, we can set up the proportion \\begin{align*} \\frac{x}{270}& =\\frac{1}{15}\\\\\\Rightarrow \\qquad x& =\\left(\\frac{1}{15}\\right)(270)\\\\\\Rightarrow \\qquad x& =\\boxed{18}\\end{align*}###
# Problem: Compute $95^2$ in your head.
# Solution: Let's think step by step. We have $(90 + 5)^2 = 90^2 + 2(90)(5) + 5^2 = 8100 + 900 + 25 = \\boxed{9025}$.###
# Problem: If $2^8=16^x$, find $x$.
# Solution: Let's think step by step. We can write $16$ as $2^4$. Therefore, we can write our equation as $2^8 = 2^{4 \\cdot x}$. Solving, we get that $x = \\boxed{2}$.###
# Problem: {problem}
# Solution: Let's think step by step.""")
HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE: str = (
"""Given a mathematics problem, determine the answer. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}.###
Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$?
Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction.
Solution: Let's think step by step. Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}. The final answer is: \\boxed{\\frac{1}{2}}.###
Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$?
Solution: Let's think step by step.Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$. The final answer is: \\boxed{0}.###
Problem: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet?
Solution: Let's think step by step. Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\\cdot800=19\\cdot200=\\boxed{3800}$ feet. The final answer is: \\boxed{3800}.###
Problem: If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$?
Solution: Let's think step by step. We know that $(x + y)^2 = (x^2 + y^2) + 2xy = 25$. We are given that $xy = 6$. So, by substitution, $x^2 + y^2 + 2xy = x^2 + y^2 + 2(6) = 25$. It follows that $x^2 + y^2 = 25 - 12 = \\boxed{13}$. The final answer is: \\boxed{13}.###
Problem: On a hot day, Megan likes to eat a Popsicle every 15 minutes. Assuming she keeps up that rate of consumption, how many Popsicles can Megan finish in 4 hours and 30 minutes?
Solution: Let's think step by step. Let $p$ be the number of Popsicles Megan can finish in 4 hours and 30 minutes. If we convert that period of time into minutes, we find that 4 hours and 30 minutes is equal to $(4)(60)+30=270$ minutes. From here, we can set up the proportion \\begin{align*} \\frac{x}{270}& =\\frac{1}{15}\\\\\\Rightarrow \\qquad x& =\\left(\\frac{1}{15}\\right)(270)\\\\\\Rightarrow \\qquad x& =\\boxed{18}\\end{align*}. The final answer is: \\boxed{18}.###
Problem: Compute $95^2$ in your head.
Solution: Let's think step by step. We have $(90 + 5)^2 = 90^2 + 2(90)(5) + 5^2 = 8100 + 900 + 25 = \\boxed{9025}$. The final answer is: \\boxed{9025}.###
Problem: If $2^8=16^x$, find $x$.
Solution: Let's think step by step. We can write $16$ as $2^4$. Therefore, we can write our equation as $2^8 = 2^{4 \\cdot x}$. Solving, we get that $x = \\boxed{2}$. The final answer is: \\boxed{2}.###
Problem: {problem}
Solution: Let's think step by step.""")
def get_math_problem_prompt_ala_helm_8shot_cot2(data_pt: dict, prompt_template: str = HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE) -> Union[str, None]:
    # Note: instead of using .format(name=value) we are using the exact string for the placeholder so that the .format function doesn't get confused with curly braces.
    return prompt_template.replace("{problem}", data_pt['problem'])

MATH_PROMPT_0SHOT_COT_TEMPLATE: str = (
"""Given a mathematics problem, determine the answer. Simplify your answer as much as possible. Think step by step, then always give the final answer inside a \\boxed{answer}
Problem: {problem}
Solution: Let's think step by step.""")
def get_math_problem_prompt_ala_0shot_cot(data_pt: dict, prompt_template: str = MATH_PROMPT_0SHOT_COT_TEMPLATE) -> Union[str, None]:
    # Note: instead of using .format(name=value) we are using the exact string for the placeholder so that the .format function doesn't get confused with curly braces.
    # inspired from the requirement for Claude 3.5 Sonnet: https://www.anthropic.com/news/claude-3-5-sonnet
    return prompt_template.replace("{problem}", data_pt['problem'])
MATH_PROMPT_0SHOT_COT_TEMPLATE_MISTRAL7B_INS_V1: str = (
"""<s>[INST] Given a mathematics problem, determine the answer. Simplify your answer as much as possible. Think step by step, then always give the final answer inside a \\boxed{answer}
Problem: {problem} [/INST]</s>
Solution: Let's think step by step.""")

HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE_MISTRAL7B_INS_V1: str = (
"""<s>[INST] Given a mathematics problem, determine the answer. Simplify your answer as much as possible. Always give the final answer inside a \\boxed{answer}###
Problem: Let $r=3^s-s$ and $s=2^n+1$. What is the value of $r$ when $n=2$? [/INST]</s>
Solution: Let's think step by step. First substitute $n=2$ into the expression for $s$ to find $s=2^2+1=5$. Then substitute $s=5$ into the expression for $r$ to find $r=3^5-5=243-5=\\boxed{238}. The final answer is: \\boxed{238}.###
[INST] Problem: If $x^{2y}= 4$ and $x = 4$, what is the value of $y$? Express your answer as a common fraction. [/INST]
Solution: Let's think step by step. Plugging $x = 4$ into the first equation, we get $4^{2y} = 4^1 \\Rightarrow 2y = 1 \\Rightarrow y = \\boxed{\\frac{1}{2}}. The final answer is: \\boxed{\\frac{1}{2}}.###
[INST] Problem: If $y = \\displaystyle\\frac{1}{3x+1}$, what is the value of $x$ when $y = 1$? [/INST]
Solution: Let's think step by step.Since $y=1$, we have $1 =\\displaystyle\\frac{1}{3x+1}$. Multiplying both sides by $3x+1$, we have $$3x+1=1$$ $$\\Rightarrow \\qquad 3x=0$$ $$\\Rightarrow \\qquad x=\\boxed{0}$$. The final answer is: \\boxed{0}.
[INST] Problem: A scale drawing of a park shows that one inch represents 800 feet. A line segment in the drawing that is 4.75 inches long represents how many feet? [/INST]
Solution: Let's think step by step. Each inch of the 4.75-inch line segment represents 800 feet, so the whole line segment represents $4.75\\times800=\\frac{19}{4}\\cdot800=19\\cdot200=\\boxed{3800}$ feet. The final answer is: \\boxed{3800}###
[INST] Problem: If $(x + y)^2 = 25$ and $xy = 6$, what is the value of $x^2 + y^2$? [/INST]
Solution: Let's think step by step. We know that $(x + y)^2 = (x^2 + y^2) + 2xy = 25$. We are given that $xy = 6$. So, by substitution, $x^2 + y^2 + 2xy = x^2 + y^2 + 2(6) = 25$. It follows that $x^2 + y^2 = 25 - 12 = \\boxed{13}$. The final answer is: \\boxed{13}###
[INST] Problem: On a hot day, Megan likes to eat a Popsicle every 15 minutes. Assuming she keeps up that rate of consumption, how many Popsicles can Megan finish in 4 hours and 30 minutes? [/INST]
Solution: Let's think step by step. Let $p$ be the number of Popsicles Megan can finish in 4 hours and 30 minutes. If we convert that period of time into minutes, we find that 4 hours and 30 minutes is equal to $(4)(60)+30=270$ minutes. From here, we can set up the proportion \\begin{align*} \\frac{x}{270}& =\\frac{1}{15}\\\\\\Rightarrow \\qquad x& =\\left(\\frac{1}{15}\\right)(270)\\\\\\Rightarrow \\qquad x& =\\boxed{18}\\end{align*}. The final answer is: \\boxed{18}###
[INST] Problem: Compute $95^2$ in your head. [/INST]
Solution: Let's think step by step. We have $(90 + 5)^2 = 90^2 + 2(90)(5) + 5^2 = 8100 + 900 + 25 = \\boxed{9025}$. The final answer is: \\boxed{9025}
[INST] Problem: If $2^8=16^x$, find $x$. [/INST]
Solution: Let's think step by step. We can write $16$ as $2^4$. Therefore, we can write our equation as $2^8 = 2^{4 \\cdot x}$. Solving, we get that $x = \\boxed{2}$. The final answer is: \\boxed{2}###
[INST] Problem: {problem} [/INST]
Solution: Let's think step by step.""")
def get_math_problem_prompt_ala_helm_8shot_cot2(data_pt: dict, prompt_template: str = HELM_MATH_PROMPT_8SHOT_COT2_TEMPLATE) -> Union[str, None]:
    # Note: instead of using .format(name=value) we are using the exact string for the placeholder so that the .format function doesn't get confused with curly braces.
    return prompt_template.replace("{problem}", data_pt['problem'])

# -- Official MATH examples from Hendryck's ref: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/math_scenario.py#L293
# MATH_PROMPT_OFFICIAL_TEMPLATE: str = (
# """Given a mathematics problem, determine the answer. Simplify your answer as much as possible.
# ###
# Problem: What is $\left(\frac{7}{8}\right)^3 \cdot \left(\frac{7}{8}\right)^{-3}$?
# Answer: $1$
# ###
# Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?
# Answer: $15$
# ###
# Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$
# Answer: $\sqrt{59}$
# ###
# Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?
# Answer: $\frac{1}{32}$
# ###
# Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?
# Answer: $181$
# ###
# Problem: Calculate $6 \cdot 8\frac{1}{3}
# Answer: $50$
# ###
# Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?
# Answer: $2$
# ###
# Problem: How many zeros are at the end of the product 25 $\times$ 240?
# Answer: $3$
# ###
# Problem: What is $\dbinom{n}{n}$ for any positive integer $n$?
# Answer: $
# """)

# -- Our OpenAI Default System Prompt
SYSTEM_PROMPT_DEFAULT: str = (
    "You are an expert mathematician. You can solve any math problem at highschool level, competition highschool level, the IMO and the prestigious Putnam Exam."
)