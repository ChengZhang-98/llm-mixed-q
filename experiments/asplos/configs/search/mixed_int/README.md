# Conditional Search

The tensor-wise mixed-precision search is a conditional search that searches the width of each GEMM op and determines the fraction width using the statistic profile. This search space is smaller than the unconditional search space which includes the fraction width.

Conditional search includes the following models:
- bert-base
- bert-large
- opt-125m
- opt-350m
- opt-1.3b
- llama-160m

Other models are searched using unconditional search (both the width and the fraction width are searched), including:
- opt-2.7b
- opt-6.7b
- llama-7b
- alpaca-7b
- vicuna-7b