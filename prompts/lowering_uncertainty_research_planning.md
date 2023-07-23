# ----  ----
references: 
    - https://web.stanford.edu/class/cs197/slides/04-vectoring.pdf 

We should have three prompts. One tackling a different way to mitigate risks and uncertainties in projects with 
uncertainties -- especially in computer science and machine learning research.
They are:
- Prompt 1: attempting to guess/extrapolate/identifying unknown unknowns -- to reduce risks we are unaware of.
- Prompt 2: asses my vectoring plan with assumptions for effective tackling of the unknown i.e., research

# Prompt1: Make a smart Inference of what the unknown unknowns are of my research plan
TODO


# Promp2: Asses my Research Plan to Tackle uncertainty under the Vectoring in Research framework/methodology
TODO
```markdown
Description of Vectoring methodology

```
my research plan:
```markdown

```
give me excellent feedback, that increases the chances I succeed and if you can idenitfy a potential blind spot that might
cause a failure or waaste my time:

# Notes

## Unknown Unknowns
start from here: https://chat.openai.com/share/e98bc26d-7f72-44b6-96ef-7173add183c5

## Vectoring in Research CS 197
```markdown
"Vectoring in Research," as presented in Stanford's CS 197 & 197C course by Sean Liu & Lauren Gillespie, refers to an approach or methodology for tackling complex research projects. Here's a summarized interpretation of the concept:

Vectoring is an iterative process used in research to identify and focus on the most critical aspect, or "dimension of risk," in a project at a given time. This critical aspect is called a "vector."

The process is not linear, where you start with an idea and then simply work towards a final result. Instead, research is an exploration where vectoring helps guide the path and prioritize tasks.

The goal of vectoring is to manage and reduce risk and uncertainty. Instead of trying to solve all aspects of a problem simultaneously, researchers pick one vector (dimension of uncertainty) and focus on reducing its risk and uncertainty within a short time frame, typically 1-2 weeks.

The concept advocates an iterative approach, where after one vector's risk is mitigated, new vectors (risks or uncertainties) might emerge. Then, the process of vectoring is repeated for the new vector. This continuous re-vectoring allows researchers to keep honing the core insights of the research project.

Different vectors might imply building different parts of your system or project, but instead of building all at once, you reduce uncertainty in the most important dimension first and then build out from there.

The vectoring process involves generating and ranking questions based on their criticality, then rapidly answering the most critical question. This approach is often supplemented by assumption mapping, which is a strategy for articulating and ranking questions based on their importance and the level of known information.

Successful vectoring enables rapid iteration cycles and a focused approach, which allows researchers to efficiently work through complex problems and avoid getting overwhelmed by trying to solve everything at once.

In summary, Vectoring in Research is a method to manage complexity and risk in research projects by focusing on the most significant risks or uncertainties, reducing them, and then moving on to the next most important risk or uncertainty, in a continuous iterative process.
```

```markdown
One sentence summary of Vectoring: 
Vectoring in research is the targeted reduction of uncertainty in the direction of project's most impactful aspect.
```
ref: gpt4 summary: https://chat.openai.com/share/f36336ec-ecf2-4e15-a130-4f3782f27baf
ref: ground truth: https://web.stanford.edu/class/cs197/slides/04-vectoring.pdf
ref: one sentence summary of vectoring: https://chat.openai.com/share/f36336ec-ecf2-4e15-a130-4f3782f27baf

## What is an optimal decision in research? i.e., projects with high uncertainty/risks
The one that most quickly moves us in the direction of our most essential goal.
GPT4 says:
> An optimal decision would be one that allows us to make significant progress towards our primary research goal by effectively addressing the most important unknowns.

```markdown
Your suggestion aligns well with the principles of Vectoring in Research. In the context of vectoring, the most optimal decision is indeed the one that rapidly reduces the risk or uncertainty in the most crucial dimension or "vector" of your project at a given time.

This is based on the principle that it's more effective to focus on resolving the most critical uncertainties first, rather than trying to address all uncertainties or complete all tasks simultaneously. As such, an optimal decision would be one that allows us to make significant progress towards our primary research goal by effectively addressing the most important unknowns.

However, it's essential to remember that "most quickly" doesn't necessarily mean rushing through steps. It emphasizes the idea of "velocity" – efficiently reducing risk in the chosen dimension, without compromising the quality of the research. The goal is to achieve a deep understanding of the most crucial aspects of your research as rapidly as possible, so that the rest of the project can be built on a solid foundation.

In this way, vectoring promotes both efficiency and depth in research, helping researchers to navigate complex projects and focus their efforts where they are most needed. The iterative nature of this process also allows for continual reassessment and adjustment, ensuring that the research remains focused and relevant as it progresses.
```

## Lean Startup methodology
TODO 

## Assumptions & Sanity Checks

```markdown
The following is why assumption checking with simple sanity checking is important.
Your most important objective is to achieve your goal under an environment of uncertainty.
If you choose to take an action, you want that action to move you forward towards that goal.
You usually base your actions based on the assumption that the place you are currently standing is correct.
If your not a correct situation, then you are basing your action on a falsehood, which will likely mean that the action you are taking will not work (since it's based on a false assumption).
Therefore, the first step must be making sure you are standing on truth. Then your action to make progress has a better chance to give you valuable information i.e., that the action you chose (with it's underlying assumption) is correct -- or not. 
Therefore, sanity check your current situation/place. Your next step should also be a sanity check that validates your new (now untested) assumption. So that at each step you are progressing you are getting closer to truth.

This is similar to Michael Bernstein's of vectoring: 
what is the most central question/uncertainty about whether your idea will work/is right? What is the easiest, 
most direct way for you to get a definitive answer to that question? Do that experiment. Rinse and repeat.

I think it's also crucial to be radically honest when making the final decision. It's very easy to unintentionally choose
an action that is not the most direct way to get a definitive answer to the most central question. For examle, it's 
happened to me that the most direct answer is slightly harder (e.g., reading through someones code) vs attempting the
seemingly easier one (e.g., a (very badly) written tutorial). But by skimming the two options one could have known that
the tutorial was badly written so it likely wasn't good -- or even better, santiy check their tutorial by running it
exactly as they wrote it. If it doesn't work you get quick feedback that the seemingly easier/quicker route is not 
actually easier because even the most basic check fails (their tutorial doesn't work on their own example). Thus, it
jumped to becoming to fix the tutorial. But the original code worked because they published a paper validating it. So,
how can we go through the original code in the easiest way as possibel? To my surprise Claude 2.0 did provide a very
reasonably sounding explanation of their code. 

Main takeaways/summary as a 4 step protocol:
1. Identify the most essential goal.
2. Stand on truth: It's crucial your starting point is based on truth e.g., do a **sanity check**. If this is not true then your next action cannot give you useful information (since you need to go back and fix what your starting point)
3. Outline most promising vectoring actions: Now that you know your standing on truth you can **outline the most promising actions** that that will give you the most information in the most essential direction
4. Articulate Assumptions: Outline a couple of actions and **articulate the assumptions** they make. You can also articulate risks/uncertainties that you are aware of.
5. Choose optimal vectoring action: Choose the optimal vectoring action i.e., the one that **most directly make progress**/teaches you the most towards your goal and has **least/lowest risk**.
Always be radically honest with yourself when choosing the actions you decided so that you really know your action is being taken for the right reason.
GPT4 summary of my text + summary:
Sanity Check: Always initiate your decision-making process with a sanity check. Ensure the validity of your assumptions 
   and the accuracy of your starting position. This will prevent actions based on falsehoods.
Identify Potential Actions: Once you've confirmed the truth of your initial position, identify potential actions that 
   could help you achieve your goal. These actions should help you gain the necessary information to progress.
Evaluate Risks and Assumptions: For each potential action, evaluate the risks and articulate the assumptions they rely on.
   Understanding the uncertainties involved in each action is crucial.
Choose the Best Course of Action: Based on your evaluation, choose the most direct and least risky action that brings 
   you closer to your goal.
Honest Self-Reflection: Always maintain radical honesty with yourself when making decisions. It's easy to 
   unintentionally choose an action that seems simpler but doesn't lead you towards the right answer.

Main takeaways/summary as a 4-step protocol (mine + GPT4s):
1: Identify the most essential goal: Identify the most essential/crucial/impactful goal/aspect/vector/direction of your project and it's uncertainties.
2. Stand on truth via a Sanity check: It's crucial your starting point/assumptions are based on truth, so that your next action can give you useful information.
   Otherwise, you might have to go back to fix it. A good way to do this is to initiate your decision-making process with a sanity check. 
   This ensures the validity of your assumptions/starting point and the accuracy of your starting position. This will prevent future actions failing due to falsehoods.
3. Identify Most Pormising Potential Actions: Once you've confirmed the truth of your initial position, identify potential 
   actions that align with the most crucial/impactful aspect of the project and how to lower it lowers uncertainty.
   It's likely useful to identify assumptions (or uncertainties) you are making about each action.
4. Articulate Risks & Assumptions: For each potential action, articulate & evalutate the risks and articulate the assumptions they rely on.
   Understanding the (hidden) uncertainties involved in each action is crucial.   
5. Choose Optimal Action: Based on your evaluation, choose the most direct and least risky/uncertain action that brings 
   you closer to your goal. Balance between directness, speed and least risky action. Maximizing for success and maximizing learning.
Honest Self-Reflection: Always maintain radical honesty with yourself when making decisions -- 
   are you really taking the most direct action that aligns with your most essential/important goal?. 
   It's easy to unintentionally choose an action that seems simpler but doesn't lead you towards the right answer.
Vectoring: Vectoring in research is the targeted reduction of uncertainty/risks in the direction of project's most crucial/impactful aspect/direction.
To avoid spending more time planning/thinking than doing, put a 2-3 limit on the actions you write down, maximize assesing in heard for speed. 

GPT4s improved summary of my improved summary:
Identify Your Essential Goal: Understand the most crucial aspect or direction of your project, along with its uncertainties. This is the goal you're striving to achieve.
Conduct a Sanity Check: Make sure your starting point and assumptions are grounded in truth. A sanity check can validate your assumptions and ensure accuracy. This prevents future actions from being based on inaccuracies or falsehoods.
Identify Promising Actions: Once you've confirmed the validity of your starting point, outline potential actions that align with your identified goal. These actions should aim to reduce uncertainty related to your project's crucial aspects. Be aware of any assumptions or uncertainties tied to each action.
Evaluate Risks & Assumptions: Articulate and evaluate the risks and assumptions associated with each potential action. Understanding the uncertainties involved in each action is key.
Choose the Optimal Action: Based on your evaluation, choose the most direct action that carries the least risk and reduces the most uncertainty, bringing you closer to your goal. Balance speed, directness, and risk management.
Practise Honest Self-Reflection: Maintain honesty with yourself when making decisions. It's easy to choose an action that seems simpler but may not lead to the right answer.
Vectoring: Vectoring is a research method that focuses on targeted reduction of uncertainty in your project's most crucial direction.
Efficient Planning: To maintain momentum and avoid excessive planning, limit your action choices to 2-3 options and try to assess them quickly.
These minor edits make your summary more clear and accessible to a broader audience.
```
ref: chatgpt: https://chat.openai.com/share/30f0e07b-7ead-4dc0-8a5f-c351c20bc178 since I created this concept, I don't have another reference. 
## Sanity Checks
TODO: ask what chat gpt thinks + this 
> Conduct a Sanity Check: Make sure your starting point and assumptions are grounded in truth. A sanity check can validate your assumptions and ensure accuracy. This prevents future actions from being based on inaccuracies or falsehoods.

## Chris Hahn's research approach: Try to break it so that what survives is the best
TODO