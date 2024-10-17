"""
ref: https://github.com/unslothai/unsloth/issues/1039
"""
from vllm import LLM, SamplingParams           # For initializing the vLLM model and sampling parameters
from vllm.lora.request import LoRARequest      # For creating a LoRA request to use in text generation

# Step 1: Specify the Path to the LoRA Adapter Checkpoint
lora_adapter_path = 'path2unsloth_lora_adapters'

# Step 2: Instantiate the Base Model with LoRA Enabled
llm = LLM(model="Qwen/Qwen2-1.5B", enable_lora=True)

# Step 3: Define Sampling Parameters
sampling_params = SamplingParams(
    temperature=0,  # Temperature controls randomness in generation. 0 means deterministic output.
    max_tokens=256, # Maximum number of tokens to generate
    stop=["[/assistant]"]  # Stop generation at this token
)

# Step 4: Define Prompts for Generation
prompts = [
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
]

# Step 5: Create a LoRA Request
# First parameter: a human-readable name for the adapter
# Second parameter: a globally unique ID for the adapter
# Third parameter: the path to the local LoRA adapter checkpoint
lora_request = LoRARequest("sql_adapter", 1, lora_adapter_path)

# Step 6: Generate Text with the LoRA Adapter
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=lora_request
)

# Step 7: Display the Outputs
for i, output in enumerate(outputs):
    print(f"Output {i + 1}:\n{output}\n")
