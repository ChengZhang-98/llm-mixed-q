# LLM-Mixed-Q

This repository contains the code for software-emulated mixed-precision quantization of LLM models.

## Setup

Conda environment is recommended. To create a conda environment, run:

```bash
cd llm-mixed-q
conda env create -f environment.yml
```

## Features

* Supported model architectures:
  + BERT
  + OPT
  + Llama

* Compatible with HuggingFace Transformers. Any checkpoints of supported model architecture from HuggingFace Transformers can be loaded, quantised, and evaluated.

* Supported quantisation:
  + Fixed-point (integer)
  + Logarithmic
  + Minifloat
  + De-normalized minifloat (DMF)
  + Block Logarithmic (BL)
  + Block Floating Point (BFP)
  + Block Minifloat (BMF)

* Supported search style:
  + Fine-grained / Coarse-grained search space.
    The search space can be specific layers, transformer bocks, or the whole model.

  + Search without/without statistic profiles. This feature is only for integer quantisation. When statistic profiles are available, the search space can only includes the width and the fraction width is determined by the statistic profile.

* Search objectives
  + Accuracy
  + Memory density
  + Frame Per Second (FPS)
  + FPS per LUT

* Search algorithms:
  + Random
  + TPE
  + NSGA-II
  + NSGA-III
  + QMC

## Example

### Search using fine-tuned accuracy

Here is an example of searching mixed-precision bert-base on SST-2 dataset.

Before search, we need to fine-tune pretrained bert-base on SST-2 dataset.

```bash
cd llm-mixed-q/experiments/asplos/fine_tune/

# run an example fine-tune script
./bert_base_sst2.sh
```

Then we can search the mixed-precision BFP quantisation, considering accuracy, memory density, and OPs per bit.

```bash
cd llm-mixed-q/experiments/asplos/mixed_bfp_co_optimize

# run an example search script, "2023_8_4" is a ckpt tag
./bert_base_sst2.sh \
  "2023_8_4" \
  ../../../experiments/asplos/configs/search/bfp_co_optimize/bert_base_sst2.toml
```

The search space is fine-grained and includes all layers of the model. The search algorithm is TPE. The objective function takes the form

```python
objective = alpha_1 * accuracy + alpha_2 * mem_density + alpha_3 * fps + alpha_4 * fps_per_lut
```

### Search using zero-shot prompt accuracy

For GPT-like model, we evaluate the accuracy on SST2 dataset using zero-shot prompt accuracy.

```bash
cd llm-mixed-q/experiments/asplos/mixed_bfp_co_optimize

./vicuna_7b_sst2.sh \
  "2023_8_4" \
  ../../../experiments/asplos/configs/search/bfp_co_optimize/vicuna_7b_sst2.toml
```

The search space is coarse-grained and performed on the Transformer block. The searched block is reused across the whole model. We don't search the positional embedding layer (RoPE) and fix it to 4-bit integer quantisation.
