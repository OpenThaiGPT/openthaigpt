# OpenThaiGPT

[![](https://img.shields.io/pypi/v/openthaigpt.svg)](https://pypi.python.org/pypi/openthaigpt) [![](https://pyup.io/repos/github/OpenThaiGPT/openthaigpt/shield.svg)](https://pyup.io/repos/github/OpenThaiGPT/openthaigpt/)

OpenThaiGPT focuses on developing a Thai Chatbot system to have capabilities equivalent to ChatGPT, as well as being able to connect to external systems and be able to retrieve data flexibly. Easily expandable and customizable and developed into Free open source software for everyone.

* Free software: Apache Software License 2.0
* Project Homepage: https://openthaigpt.aieat.or.th
* Documentation: https://openthaigpt.readthedocs.io.

## Features

* You can now select the model_name as follows:
* kobkrit/openthaigpt-0.1.0-alpha
* kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.4
* kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.3
* kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.2
* kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.1

## Installation
Python>=3.6

### CPU-Only
``$ pip install openthaigpt torch --extra-index-url https://download.pytorch.org/whl/cpu``

### GPU

CUDA 11.6
``$ pip install openthaigpt torch --extra-index-url https://download.pytorch.org/whl/cu116``

CUDA 11.7
``$ pip install openthaigpt torch``

## Usage
```
import openthaigpt

print(openthaigpt.generate("Q: อยากลดความอ้วนทำไง\n\nA:"))
print(openthaigpt.zero("การลดน้ำหนักเป็นเรื่องที่ต้องพิจารณาอย่างละเอียดและรอบคอบเพื่อให้ได้ผลลัพธ์ที่ดีและมีประสิทธิภาพมากที่สุด"))
```

## Using 0.1.0-alpha model
```
import openthaigpt

print(openthaigpt.generate(instruction="แปลภาษาอังกฤษเป็นภาษาไทย", input="We want to reduce weight.", model_name = "kobkrit/openthaigpt-0.1.0-alpha", min_length=50, max_length=300,  top_k=20, num_beams=5, no_repeat_ngram_size=20, temperature=1, early_stopping=True))
```

## Collaboration By
* Artificial Intelligence Entrepreneur Association of Thailand (AIEAT)
* Artificial Intelligence Association of Thailand (AIAT)

## Supported By
* NECTEC
* iApp Technology
* Pantip
* NVIDIA
* Microsoft
* Mahidol University
* Gitbook