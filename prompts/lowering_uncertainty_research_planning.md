# ---- Prompts for Assessing & Rubber Ducking discussions on your research plan according to Vectoring  ----
references: 
    - https://web.stanford.edu/class/cs197/slides/04-vectoring.pdf
    - my public gist: https://gist.github.com/brando90/00e5e3c66f5349a0c3cbc62ef2501904

We should have three prompts. One tackling a different way to mitigate risks and uncertainties in projects with 
uncertainties -- especially in computer science and machine learning research.
They are:
- Prompt 1: attempting to guess/extrapolate/identifying unknown unknowns -- to reduce risks we are unaware of.
- Prompt 2: asses my vectoring plan with assumptions for effective tackling of the unknown i.e., research

# Prompt1: Make a smart Inference/Smart Guess of what the unknown unknowns are of my research plan

# Prompt2: Assuming there will always be risks we cannot take into account (unknown unknowns), is our research plan adaptable, flexible and robust?

# Prompt3: Asses my Research Plan to rackle uncertainty under the Vectoring in Research framework methodology
You are a research expert, top CS professor at Stanford, giving feedback to make a research plan excellent. 
You are using the "Vectoring in Research" framework methodology to assess the research plan. 
In particular, with emphasis in identifying the most impactful & important research goal with future actions that 
most directly attain solve the essential goal -- while minimizing & identifying potential risks, assumptions and uncertainties. 
In more detail, this is the vectoring methodology: 
```
<vectoring>
Vectoring is an iterative process used in research to identify and focus on the most critical aspect, or "dimension of risk," in a project at a given time. 
This critical aspect or dimension is called a "vector". 
The goal of vectoring is to manage and reduce risk and uncertainty & make make most direct progress as effortlesly as possible in the most essential goal. 
Instead of trying to solve all aspects of a problem simultaneously, researchers pick one (essential) vector (dimension of uncertainty) 
and focus on reducing its risk and uncertainty within a short time frame, typically 1-2 weeks (but if shorter is possible great).
The methodology advocates an iterative approach, where after one vector's risk is mitigated, new vectors (risks or uncertainties) might emerge. 
Then, the process of vectoring is repeated for the new vector that makes most direct progress on the essential goal. 
This continuous re-vectoring allows researchers to keep honing the core insights of the research project.
The vectoring process involves generating and ranking questions based on their criticality, then rapidly answering the most critical essential question. 
This approach is often supplemented by assumption mapping, which is a strategy for articulating and ranking questions based on their importance and the level of known information.

In summary: Vectoring in Research is a method to manage complexity and risk in research projects by focusing on the most significant risks or uncertainties, 
reducing them, and then moving on to the next most important essentual risk uncertainty or goal, in a continuous iterative process.
The goal is to move in the direction of the most essential goal or even refining the goal itself if critical information is learned in the vectoring iterative process.
</vectoring>
```
Now I will provide a research plan that has already identified some potential assumptions problems and potential actions to solve mitigate them. 
The research plan will be given in the following flexible/approximate format:
```
<plan_format> 
- Plan1:
    - Essential/Impactful Goal (EssG):
    # -- Outline the Hypothesis (question) to clarify and the Assumption we are making for believing this hypothesis
    - Hypothesis & Assumption 0 (HAss0):
      # -- Then the most direct action(s) to address the most essential goal via the hypothesis that has assumptions
      - Action 0 (Ac0): ...
      - Action 1 (Ac1) : ...
      - ...
      - Decision (what & why) (Dec): ... 
      - ...
    - Hypothesis & Assumption 1 (HAss1):
      - Action 0 (Ac0): ...
      - ...
    - Hypothesis & Assumption 2 (HAss2):
      - Action 0 (Ac0): ...
      - ...
</plan_format> 
```
This my concrete plan to evaluate under the vectoring framework methodology and give feedback: 
```
<current_plan> 
</current_plan> 
```
Now I recall the vectoring methodology framework again in one sentence, since the insightful research feedback should be based on this framework: 
```
<vectoring_summary>
Vectoring in research is the targeted reduction of uncertainty in the direction of project's most impactful aspect/goal.
</vectoring_summary>
```
Now give me excellent yet concise feedback on my research plan under the vectoring methodology. 
Your feedback should increase the chances I succeed in learning about my most essential research goal. 
If you identify a potential blind spot (e.g., unknown unknowns) that might cause a failure or waste my time, mention them. 
But also remember that unknown unknowns are by definition unpredictable, so if you can't identify those blind spots then
make sure the plan is adaptable, flexible and robust.
Also feel free to suggest sanity checks to validate my starting point 
e.g., it's good to test and validate the approach on a small sample first before committing fully, due to the uncertainties. 
If the decision to take is already excellent under the vectoring methodology you can say why and confirm it's good. 
Provide helpful excellent feedback according to the specifications.

Also, the attached pdf research paper(s) and code is the background knowledge for the research project.

...

Now proceed to give me concrete feedback using the unknown unknown criteria. 
Can we try to identify unpredictable issues? 
And also, assess the flexibility, adaptability and robustness of my initial research plan to address unknown unknowns:

concrete examples: 
1. https://chat.openai.com/share/921c9bf5-e6e4-4cc5-bc8a-dbbaf2f86b42, https://claude.ai/chat/3d227faf-037e-4ec3-a220-e0390796df7f
   2. seems claude 2.0 seemed to give a better answer than gpt4, e.g., it suggested to sanity check my starting point.
2. https://chat.openai.com/share/629602aa-40a2-4225-8817-33b52417279d, https://claude.ai/chat/a4acaa59-f15c-486c-8145-e653eacfd094
   2. this time gpt4 gave a better answer, it was more detailed, Claude 2.0 seemed to short and not nuanced/insightful enough. But both provided similar and useful comments when prompted to specifically talk about unknown unknowns. e.g., quick iterative testing + no long term commitments. 
3. https://chat.openai.com/share/379bb0ab-c107-4410-9e10-a888fc0b6c18, https://claude.ai/chat/edecf623-0634-4641-9ed8-1d230402e7e3
   3. I still like claude more. Not sure why, it's nice to talk to it. GPT4 did say to articulate ess goal why its impactful, thats useful even if not written down a good reminder. I like prompting it for unknown unknowns as follows. Good responses. 
4. with background of work
   4. backgorund 1 paper pdf + 4 python files: https://claude.ai/chat/07767e4e-9992-43a1-8542-d7deb714baf0 (idk of a good gpt4 plugin that receives files, it's ok to not do that one imho since alpaca says claude is nearly as good as gpt4 and I've observed they are very comparable not really replacable since they behave differently)
   4. no background: https://claude.ai/chat/2d51712e-91d5-49aa-9184-d21b96f5713b, https://chat.openai.com/share/089fcaeb-40bd-4584-bf2f-03dfc59426ed

# Adding pdfs to paper - back of the envelope calc
Q: Can we add the background knowledge to these discussions with Claude/ChatGPT?
Q: Interesting! I just checked out one of the ChatGPT demos. I wonder how much domain knowledge the model needs about your research area to be helpful — or if it's most useful as an elicitation device, and doesn't actually need to know anything about your topic.
Q: putting synonyms for prompt good?
Q: its better prompt engineering to have unknown unknowns seperately from the entire vectoring prompt (todo: read more about prompt engineering)

8bits = 1 bytes = 1 char --> 
Claude allows 5 files 10MB each. 
My PT vs MAML paper is 27 pages, 11 figures 23 tables = 2.4MB long.
So we can fit 5 real pdfs in the prompt to claude for vectoring advice + likely merge more papers into the pdfs to reach the 10MB limit.

# Evaporate summary

```markdown
Evaporate is a system that uses large language models (LLMs) to extract structured data from unstructured or semi-structured documents.

The key components I see are:

Evaporate-Direct directly prompts the LLM to extract attributes and values from each document. This provides high quality but is expensive as it processes every document.
Evaporate-Code synthesizes Python functions using the LLM to extract attributes, then applies these functions to process documents. This reduces cost but can have lower quality.
Evaporate-Code+ improves on Code by generating multiple candidate functions and aggregating their outputs using weak supervision. This aims to improve quality while maintaining low cost.
The system supports different document formats like HTML, PDF, text without customization by prompting the LLM.
It evaluates on tasks like extracting tables from scratch (open IE) as well as populating a predefined schema (closed IE).
I will provide the code. The code handles chunking large documents, scoring and selecting generated functions, and aggregating multiple noisy extractions. 
The prompts provide examples for the LLM. Overall, it's a novel approach to information extraction that leverages LLMs' few-shot learning capabilities. 
```

# Notes

## Unknown Unknowns

```markdown
Unknown unknowns are risks that we are unaware of.

"Known unknowns" and "unknown unknowns" are terms used in risk management and project planning. 
They refer to different types of uncertainties that one might encounter in a project or task. 
Understanding these concepts can help in identifying, managing, and mitigating risks effectively.

Known Unknowns: These are uncertainties or risks that we are aware of but do not know the outcome.
In risk management & project planning, these risks are often identified during the planning phase and are accounted for 
by building buffers or safety margins that try to account for the known unknown (known uncertainty).
These potential problems (known unknown) are identified, and plans are made to help deal with them. 
For example:
Time Contingency/Buffer: Extra time is allocated in the project schedule to manage potential delays.
Cost/Financial Contingency/Buffer: Extra budget is allocated to handle potential cost overruns.
Resource Contingency/Buffer: Extra resources (such as personnel or equipment) are planned for in case they are needed. 
Plan B/Backup Plan: Sometimes, an alternative approach or strategy (a "Plan B") is developed and can be put into action if a particular risk materializes.

Unknown unknowns: These are risks or uncertainties that we're not even aware exist, so we cannot plan for them.

```

```markdown
Known unknowns:
Known Unknowns: These are uncertainties or risks that we are aware of but do not know the outcome. For example, we might 
know that a particular phase of a project will have challenges because we are using a new technology or method, but we 
don't know what specific issues might arise or how severe they will be. This uncertainty is 'known' because we're aware 
of its existence, even though we don't know the details or implications. In project management, these risks are often 
identified during the planning phase and are accounted for by building in contingencies, such as additional time or 
resources, to manage them.

Certainly, another term often used instead of "contingencies" is "buffer" or "safety margin". These terms all refer to the extra resources — time, money, personnel, etc. — that are allocated in a project plan to handle potential risks or uncertainties, or the "known unknowns."

So, when we say that "risks are often identified during the planning phase and are accounted for by building in buffers", it means that during the planning phase of a project, potential problems are identified, and plans are made to help deal with them. These plans usually involve allocating extra resources (the buffer) to manage these potential problems if they do arise.

For example:

Time Buffer: Extra time is built into the project schedule to manage possible delays. This means that if a task is estimated to take ten days, but there's a chance it might be delayed, the project schedule might allow for fifteen days for that task to provide a time buffer.

Financial Buffer: Additional funds are set aside in the project budget to handle unexpected costs. This means if a part of the project is estimated to cost $10,000, but there's a risk that the cost could go over that amount, a financial buffer of an additional $2,000 might be added to the budget.

Resource Buffer: Additional resources (like extra personnel or equipment) are arranged to be available in case they are needed. For instance, if a project relies on a certain piece of machinery, having a backup machine available creates a resource buffer.

Backup Plan: Sometimes, an alternate plan or strategy is prepared to be implemented if a particular risk happens. This backup plan acts as a buffer strategy to ensure the project can continue even when faced with certain issues.

By identifying potential risks and building in these buffers, project managers attempt to lessen the impact of known unknowns, thereby improving the chances of successfully completing the project within its deadline, budget, and quality requirements.
```

```markdown
Unknown unknowns: Unknown Unknowns: These are risks or uncertainties that we're not even aware exist, so we cannot plan 
for them. 
For instance, during a construction project, an unknown unknown might be an archaeological discovery on the 
construction site that halts work. These are unpredictable risks that were not foreseen and could not have been 
reasonably predicted. 
**Since we can't identify these risks in advance, we can't plan for them directly. 
Instead, we  manage these risks indirectly by ensuring that our project plans are flexible and adaptable, 
and that we have the capacity to respond effectively to unexpected events when they occur**.

Sure, let's use an example from a field like pharmaceutical research and development (R&D), where dealing with unknown unknowns is quite common.

Suppose a pharmaceutical company is conducting R&D for a new drug to treat a specific disease. They've planned for known unknowns such as potential difficulties in synthesizing the drug, possible side effects, and regulatory hurdles, among other things.

An unknown unknown in this context might be the sudden emergence of a research study that demonstrates a certain common ingredient used in the formulation of the drug has serious health implications that were not previously known. This is something the company could not have anticipated or planned for directly, because it wasn't a known risk at the outset of the project.

The company can mitigate the risk of this unknown unknown by having an adaptable, flexible plan in place. Here's how:

Diversification of Research: Instead of putting all their resources into the development of one drug, the company can conduct parallel research on multiple potential drug formulations. This way, if there's an unexpected issue with one, they have others they can pivot to.

Regular Monitoring and Learning: The research team should regularly review the latest scientific literature and developments in their field. This keeps them updated and prepared to adapt their research based on new knowledge that emerges.

Building in Time and Budget Flexibility: The research plan should allow for unexpected developments. This might involve budgeting extra funds for unexpected costs and planning a timeline that allows for potential delays.

Stakeholder Communication: Regular communication with stakeholders (like investors, regulatory bodies, etc.) helps in managing their expectations. It also ensures that any required change in direction due to unforeseen circumstances can be better understood and accepted.

Robust Experimentation and Validation Process: By thoroughly testing and validating their results at every step, they can mitigate the impact of unexpected issues and adapt their research as required.

Iterative Development: By breaking down their research and development process into smaller, iterative steps, the company can incorporate new learnings more easily and adapt their direction more quickly if required.

Remember, the key to dealing with unknown unknowns is not to try to predict them, but to build systems and processes that allow for flexibility and adaptability in the face of unexpected events.



Let's consider a research project in computer science focusing on the development of a new machine learning algorithm. The team has identified and planned for several known unknowns, such as the difficulty in training the model, the availability of quality training data, computational resources, etc.

An example of an unknown unknown could be the discovery of a critical vulnerability in a widely used software library that the algorithm relies on, causing it to behave unpredictably or making it exploitable. This is something the team couldn't have anticipated because it wasn't a known risk when they started the project.

To mitigate the risk of such unknown unknowns, the team can employ several strategies:

Modular Design: Design the algorithm and the overall software in a modular manner. This makes it easier to replace or update individual components (like the vulnerable software library) without affecting the entire system.

Regular Updates and Review: Maintain an awareness of current developments and updates in the field of machine learning and the software libraries being used. This allows the team to quickly respond to any newly discovered issues.

Version Control: Use version control systems to keep track of changes in the codebase. This allows the team to revert to a previous state if something breaks, providing a safety net against unexpected issues.

Robust Testing: Implement comprehensive testing strategies including unit tests, integration tests, and system tests to ensure that all parts of the software are working as expected, and to catch any unexpected behaviors as early as possible.

Flexibility in Planning: Incorporate flexibility in the project timeline and budget to account for unexpected roadblocks or challenges.

Cross-functional Teams: Having a team with diverse skills can help in managing unexpected challenges. For instance, having team members with expertise in cybersecurity can help in quickly addressing a newly discovered software vulnerability.

Remember, the key to managing unknown unknowns is flexibility and adaptability, rather than trying to predict all possible issues in advance.



Sure, let's consider a scenario where a research group is working on the development of a machine learning model to predict certain outcomes based on large volumes of data. The team is aware of and has planned for many known unknowns, such as the quality of the training data, the potential for overfitting or underfitting, the choice of the most suitable algorithms, etc.

An example of an unknown unknown in this context could be the discovery of a fundamental flaw in the statistical assumptions that the model was built upon. This flaw could lead to systematically biased predictions, which was not anticipated when the project started, making it an unknown unknown.

Here are some strategies to indirectly prepare for such unknown unknowns:

Thorough Initial Research: Prior to embarking on the model development, ensure a thorough literature review and understanding of the problem domain. This can provide early insights into potential pitfalls or alternate methodologies that might be worth considering.

Ensemble Methods: Instead of relying on a single predictive model, use ensemble methods that combine the predictions of several models. This can help improve prediction accuracy and robustness, and it might mitigate the impact of any one model's assumptions being off.

Regular Cross-validation: Perform regular cross-validation checks during the model development phase. This not only helps in tuning the model parameters but could also highlight any fundamental inconsistencies or flaws in the model's predictions.

Adaptable Architecture: Develop the model in such a way that key assumptions or components can be modified without requiring a complete overhaul. This gives the team the ability to adapt the model based on new learnings or findings.

Broad Skillset in the Team: A team with a diverse set of skills could bring different perspectives, potentially spotting issues that might not be apparent if everyone had the same background.

Peer Reviews and Collaboration: Encourage frequent peer reviews and collaborations with other researchers in the field. This exposes your methods and assumptions to more scrutiny, increasing the chances of uncovering potential unknown unknowns.

Remember, the key here is not to predict the unknown unknowns but to develop a system that is adaptable and resilient enough to handle unexpected events when they arise.



Sure, let's consider a deep learning research project that involves the development of a large language model based on transformers. The research team would be aware of, and plan for, several known unknowns, such as the computational resources required for training, the quality and quantity of training data, the choice of hyperparameters, etc.

An example of an unknown unknown in this context could be a discovery that certain linguistic or cultural biases in the training data lead to a significant impact on the performance of the model in certain contexts, beyond what was initially expected or understood. This issue, not being fully understood or predictable at the start of the project, qualifies as an unknown unknown.

To mitigate the potential impact of such unknown unknowns, the research team could consider the following strategies:

Comprehensive Data Review: Ensure a thorough review and understanding of the training data, including its potential limitations and biases. This knowledge might help to identify and address previously unconsidered issues.

Model Interpretability: While interpretability in deep learning is challenging, techniques such as attention visualization, saliency maps, or LIME (Local Interpretable Model-Agnostic Explanations) could be used to better understand how the model is making its predictions.

Regular Evaluation and Validation: Conduct frequent and diverse evaluation of the model's performance across various contexts and datasets. This might include not just standard benchmarks, but also novel scenarios or edge cases.

Diversity in Research Team: A research team that is diverse in terms of linguistic and cultural backgrounds might be better placed to identify unexpected biases or performance issues.

Iterative Development: Use an iterative development approach, where feedback from each stage of testing and evaluation is used to refine the model, its training process, or the data it is being trained on.

External Collaborations and Reviews: Engaging with the broader research community for peer reviews, collaborations, or shared evaluations can provide additional perspectives and scrutiny, potentially highlighting unknown unknowns.

Remember, the goal is not necessarily to predict unknown unknowns (as by definition, they can't be predicted), but to create a research and development process that is robust and flexible enough to respond effectively when unexpected issues do arise.


Certainly, let's consider a deep learning project where the goal is to develop a transformer-based language model for advanced natural language understanding tasks. This team has planned for known unknowns such as computational requirements, model architecture choices, and hyperparameter tuning.

An example of an unknown unknown could be the discovery that the transformer model behaves unpredictably when dealing with a certain complex linguistic construct, one that wasn't part of the initial testing and validation scenarios. The model's inability to generalize to these constructs wasn't known or predicted at the project's outset, making it an unknown unknown.

Here are several strategies to indirectly prepare for such unknown unknowns:

Diverse Dataset Creation: While it's not directly about bias, ensuring a diverse range of linguistic constructs and languages in your training data can potentially uncover unexpected behaviors of your model.

Continual Validation and Testing: Regularly test the model against a variety of different scenarios, including complex and unusual linguistic constructs or contexts.

Model Debugging Techniques: Utilize techniques and tools designed for debugging deep learning models. Tools like TensorBoard can help visualize the internal workings of a model and could potentially highlight issues not initially foreseen.

Modularity in Model Architecture: Construct the model in a modular way so that parts of the architecture can be swapped or altered as needed. If an unknown unknown arises that requires a change in architecture, this modularity can allow for easier adaptation.

Continual Learning: Employ continual learning strategies, where the model can adapt and learn from new data or scenarios even after the initial training phase. This could potentially help the model adapt to unanticipated scenarios or constructs.

Engagement with the Research Community: Stay engaged with the wider research community. New papers, techniques, and findings can shed light on potential unknown unknowns.

Ensemble Approaches: Using a collection of models instead of just one can improve robustness. If an unknown unknown affects one model in the ensemble, the others might still provide reliable outputs.

Again, the primary objective here isn't to predict all possible unknown unknowns, but to develop an adaptable and resilient research process to manage unexpected issues as they arise.
```

start from here: https://chat.openai.com/share/e98bc26d-7f72-44b6-96ef-7173add183c5

## Vectoring in Research CS 197`
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
Vectoring in research is the targeted reduction of uncertainty in the direction of project's most impactful/essential aspect/goal.
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

## Tips

- Quick Skimming: I learned one can asses a direction quickly without delving to much in depth to it by skimming the options e.g., read the tutorial, does it actually look useful/good/quick/high quality?

## Concrete examples of research assesment

```markdown
Asses my Research Plan to Tackle uncertainty under the Vectoring in Research framework/methodology You are an top research expert professor at Stanford giving feedback. I'd like to asses my research plan under the "Vectoring in Research" framework/methodology. In particular, with emphasis in identifying the most important goal along with future actions to reach that goal in the most direct way possible, while considering potential risks, assumptions and uncertainties. In more detail, this is the vectoring methodology: ``` Vectoring is an iterative process used in research to identify and focus on the most critical aspect, or "dimension of risk," in a project at a given time. This critical aspect is called a "vector." The goal of vectoring is to manage and reduce risk and uncertainty. Instead of trying to solve all aspects of a problem simultaneously, researchers pick one vector (dimension of uncertainty) and focus on reducing its risk and uncertainty within a short time frame, typically 1-2 weeks. The concept advocates an iterative approach, where after one vector's risk is mitigated, new vectors (risks or uncertainties) might emerge. Then, the process of vectoring is repeated for the new vector. This continuous re-vectoring allows researchers to keep honing the core insights of the research project. The vectoring process involves generating and ranking questions based on their criticality, then rapidly answering the most critical question. This approach is often supplemented by assumption mapping, which is a strategy for articulating and ranking questions based on their importance and the level of known information. In summary, Vectoring in Research is a method to manage complexity and risk in research projects by focusing on the most significant risks or uncertainties, reducing them, and then moving on to the next most important risk or uncertainty, in a continuous iterative process. The goal is hopefully moving in the direction of the most essential goal or even refining the goal itself if critical information is learned in the vectoring iterative process. ``` Now I will try to identify a few potential actions with their assumptions (uncertainties) asess it under the vectoring framework. This is the format: ``` <plan_format> Essential/Impactful Goal (EssG): Plan: Action 1 (Ac1): Assumption 1 (Ass1): Action 2 (Ac2): Assumption 2 (Ass2): Action 3 (Ac3): Assumption 3 (Ass3): My chosen decision of which action to take and why: </plan_format> <plan> - plan - EssG: extract informal thms/pfs and put into a data frame with informal thms/pfs - Ac1: use original evaporate to get i thms/pfs -> i df via Claude 2.0 or Copilote Chat - Ass1: evaporate is a published research by stanford so it's likely works - Ass2: easy to figure out how to use despite little code - Decision: - I actually don't know of a different action that is more direct with least uncertainty than Ac1 right now, that doesn't require me to think of complete alternatives/pivots e.g., extract them with an LLM or re-implementing evaporate or find better work than evaporate. Right now this also leverages that I'm familiar with evaporate. </plan> ``` Now I will recall vectoring in once sentence before I request for feedback on my research plan: ``` Vectoring in research is the targeted reduction of uncertainty in the direction of project's most impactful aspect/goal. ``` No give me excellent yet concise feedback on my research plan under the vectoring methodology. Your feedback should increase the chances I succeed in learning about my most essential goal. If you idenitfy a potential blind spot (e.g., unknown unknowns) that might cause a failure or waaste my time, mention them. If the decision to take is already excellent under the vectoring methodology you can say why and confirm it's good. Provide helpful feedback:


Claude 2.0:
Here is my feedback on your research plan under the vectoring methodology:

Your essential goal of extracting informal theorems/proofs into a dataframe seems quite reasonable as an initial step. Action 1 of using the existing evaporate tool to extract these informal results is a good starting point, as it leverages prior work and your familiarity with the tool. 

The main assumption around evaporate working as intended seems fair, since it is published research. However, I would suggest testing it out on a small sample first to validate your assumptions, rather than assuming it will work smoothly at scale right away. 

Overall your proposed action seems well aligned with vectoring principles - it focuses on a clear essential goal, proposes a direct action to make progress, and names the key assumption. My main suggestion would be to test and validate the approach on a small sample first before committing fully. This helps reduce uncertainty and risk.

Additionally, some potential blind spots to consider:

- Will the output format of evaporate need cleanup/processing to get it into the desired dataframe format? Factor in time for this.

- How will you evaluate quality of the extracted theorems/proofs? Garbage in, garbage out.

- Can you access or sample the informal source content needed to test evaporate at small scale first?

But overall this seems like a reasonable starting point. The key will be to start small, validate assumptions early, and be ready to pivot if needed. Let me know if this helps provide useful vectoring feedback on your proposed research plan!
```

## Velocity

```markdown
In the context of vectoring in research, velocity refers to the speed and efficiency with which you address and reduce the risk or uncertainty in the chosen dimension (or vector). Essentially, it's about how quickly you can move towards achieving your goals in a particular area of your research.

The main idea behind the concept of velocity is to focus your efforts and resources on rapidly answering the most critical question or resolving the most pressing issue identified during the vectoring process. By doing so, you can minimize wasted time and effort on less critical aspects and ensure that your research moves forward in a meaningful and efficient way. This "tight loop" of vectoring and velocity is essential for successful research projects.

Keep in mind that velocity in this context doesn't refer to rushing through research haphazardly. Rather, it emphasizes strategic focus, efficient execution, and rapid learning, enabling researchers to iterate quickly and make progress on the most significant dimensions of their work.
```

ref: gpt4 summary https://chat.openai.com/share/e090a6cc-26e2-470e-adda-afd7019ea8f2
ref: MB slides https://web.stanford.edu/class/cs197/slides/04-vectoring.pdf 


## Learning 

```markdown
amazing it was able to identify something I only suspected might be true after going to ICML

Now proceed to give me concrete feedback using the unknown unknown criteria. Can we try to identify unpredictable issues? And also, assess the flexibility, adaptability and robustness of my initial research plan to address unknown unknowns:
BM
You raise an excellent point about trying to identify potential unpredictable issues and assess the flexibility of the research plan. Here is some feedback along those lines:
Unknown unknowns:
A core assumption seems to be that cleaner theorem data will enable training the IL<->FL translator. But there could be other unknown gaps between the informal theorems and formal logic that cause issues, even with clean data. For example, ambiguities in natural language or different levels of rigor.
The extraction functions themselves may have inherent limitations that are currently unknown, only discovered through deeper inspection. There could be edge cases or nuances of mathematical notation that the functions cannot handle properly.


In particular, in ICML they trained a translator just on unpaired examples to translate and it was able to do it. So perhaps the clean data I'm trying to get is not needed...

I can't detect giving the paper pdf and 4 python files makes a difference.
```

```markdown
Active Learning Approach: Consider an active learning approach, where your model is continuously learning and updating itself as it processes more data. 
This can help it to adapt to any unknown unknowns that may emerge in the data over time.
```
