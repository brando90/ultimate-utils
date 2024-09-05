"""
tldr; train an inference efficient model with more tokens to get better model. Counter to Chinchilla.

Note:
    - LLaMA 6.7B=6.7e9 was trained on 1.0T=1.0e12 tokens (stated for intuition).
    - LLaMA 13.0B=13.0e9 was trained on 1.0T=1.0e12 tokens (stated for intuition).
    - LLaMA 32.5B=32.5e9 was trained on 1.4T=1.4e12 tokens (stated for intuition).
    - LLaMA 65.2B=65.2e9 was trained on 1.4T=1.4e12 tokens (stated for intuition).

ref:
    - paper: https://arxiv.org/abs/2302.13971
    - blog: https://ai.facebook.com/blog/large-language-model-llama-meta-ai/
"""