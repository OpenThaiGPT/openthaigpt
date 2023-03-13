# OpenThaiGPT

[![](https://img.shields.io/pypi/v/openthaigpt.svg)](https://pypi.python.org/pypi/openthaigpt) [![](https://pyup.io/repos/github/OpenThaiGPT/openthaigpt/shield.svg)](https://pyup.io/repos/github/OpenThaiGPT/openthaigpt/)

OpenThaiGPT focuses on developing a Thai Chatbot system to have capabilities equivalent to ChatGPT, as well as being able to connect to external systems and be able to retrieve data flexibly. Easily expandable and customizable and developed into Free open source software for everyone.

* Free software: Apache Software License 2.0
* Project Homepage: https://openthaigpt.aieat.or.th
* Documentation: https://openthaigpt.readthedocs.io

## Versions

- Code version: 0.0.8
- Model version: 0.0.4
  - Pretraining Model: GPT-2 Thai-base
  - InstructDataset: 300,000 Pantip + 5,000 Wiki QA => 12,920 Thai InstructGPT
  - Reinforcement Learning from Human Feedback (RLHF): None

## Installation
Python>=3.6

### CPU-Only
``$ pip install openthaigpt torch --extra-index-url https://download.pytorch.org/whl/cpu``

### With GPU

CUDA 11.6
``$ pip install openthaigpt torch --extra-index-url https://download.pytorch.org/whl/cu116``

CUDA 11.7
``$ pip install openthaigpt torch``

## Usage
```
import openthaigpt

print(openthaigpt.generate("Q: อยากลดความอ้วนทำไง\n\nA:"))
```
