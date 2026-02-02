# DAWN: Dependency-Aware Fast Inference for Diffusion LLMs

We propose DAWN, a training-free, dependency-aware decoding method for fast dLLM inference.

![overview](asset/overview.png)

## Project Structure

```
.
├── dream/          # Dream model related code
├── llada/          # LLaDA model related code
└── .gitignore      # Git ignore configuration
```

## Environment
- Python 3.10.12
- NVIDIA GPU + CUDA 12.1 compatible driver

## Installation
### Option A: Standard install (recommended)
```bash
pip install -r requirements.txt
```
### Option B: Reproducible install
```bash
pip install -r requirements-lock.txt
```

## Eval

We provide the eval scripts for the main experiment, you can reproduce it directly. For example:
```bash
cd llada
bash eval_instruct.sh
```
The main result:
![main result](asset/main_result.png)

## Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada), [Dream](https://github.com/dream-project/dream) and [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) for their excellent work and open-source contributions.
