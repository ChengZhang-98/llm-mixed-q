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

  + Search without / without statistic profiles. This feature is only for integer quantisation. When statistic profiles are available, the search space can only includes the width and the fraction width is determined by the statistic profile.
