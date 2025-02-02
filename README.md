# LocalGeoLite

**LocalGeoLite** is a GIS AI assistant powered by a local large language model, designed to generate GIS-related code and answer questions! 

## Features  

- 🌍 **GIS-Focused** – Generates GIS-related code and answers questions  
- 🚀 **Runs Locally** – Fast responses without relying on external servers  
- 💻 **Command-Line Interface** – Simple and interactive usage  
- 🔒 **Secure & Private** – No internet connection required, keeping your data safe  
- 🎯 **Dual Modes** – Supports both code generation and text generation
- ⚡ Zero Setup – Get started quickly with minimal requirements
## Quick start
### ⚠️ IMPORTANT
```
This project is for learning purposes and did not tested across multiple devices. 
```
### 💻 Hardware Requirements
The project only works with NVIDIA GPUs that support CUDA. 
- NVIDIA GPU with CUDA support
- Minimum 6GB GPU VRAM
- CUDA version >= 11.6 

**Note**: It is recommended to use a new virtual environment to avoid dependency conflicts. Also, make sure **Python version is 3.10 or higher**.
### Install PyTorch  

Install the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) based on your CUDA version.

**Example**: for CUDA >= 12.6 and Windowes OS

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
### Install LocalGeoLite

`pip install git+https://github.com/Junjie-Mu/LocalGeoLite.git`

For the first time the model and parameters will be downloaded locally, which may take some time. 
## Usage

There are two ways to run LocalGeoLite:

### CLI Mode 
#### Recommended when running in an IDE or terminal.

`localgeo loadmodel [cache_dir]`: The parameter **`cache_dir`** specifies the path to the model cache directory (optional). 
  ```bash
  >>>  localgeo loadmodel
  ```
`code <prompt>`: Generate GIS-related Python code
  ```bash
  >>> code Read a GeoJSON file into a GeoDataFrame
  ```
`text <prompt>`: Answer GIS-related questions
  ```bash
  >>> text UTM projection
  ```
`unload / Ctrl+C`: Exit the program.

![描述文本](https://github.com/user-attachments/assets/194e0472-e42e-4ba2-a903-826d1dadda34)

### Python API 
#### Recommended when using Jupyter Notebook.

Import the Library:
```python
from LocalGeoLite import load_model, code, text, unload_model
```
Load model:
```python
# The parameter `cache_dir` specifies the path to the model cache directory (optional). 
load_model(cache_dir="D:/Models")
```
Functions:
```python
# Generatea code
code("Read a GeoJSON file into a GeoDataFrame.");
# Generatea text
text("Descirbe TIN");
# unload model
unload_model()
```
## Model and Finetune
I built **LocalGeoLite** using **Qwen2.5-7B** 
as the base model and curated a fine-tuning dataset 
with 2000+ high-quality GIS-specific prompt-answer pairs. 
The fine-tuning process was accelerated using [**Unsloth**](https://unsloth.ai/), 
which significantly reduced GPU memory usage while improving both code generation
and question-answering capabilities for GIS tasks.

The fine-tuned model is hosted on [**Hugging Face**](https://huggingface.co/) 
and efficiently loaded and inferred using **Unsloth**. 
I further optimized the environment setup, model deployment, 
and interaction logic, packaging everything into a Python library
for a plug-and-play GIS AI assistant.

For reproducibility and further customization, the fine-tuning dataset
and training scripts are organized in the folder [Finetune](https://github.com/Junjie-Mu/LocalGeoLite/tree/main/Finetune).

## References 

- [Qwen2.5](https://github.com/QwenLM/Qwen) – LLM and LMM series of the Qwen Team, Alibaba Group.
- [Unsloth](https://unsloth.ai/) – Makes finetuning LLM faster, use 70% less memory, and with no degradation in accuracy.
- [Hugging Face](https://huggingface.co/) – Model hosting and deployment platform
