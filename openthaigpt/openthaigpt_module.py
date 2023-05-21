"""Main module."""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers.models.mt5 import MT5Tokenizer
from transformers import DataCollatorForLanguageModeling
from peft import PeftModel, PeftConfig
from evaluate import load
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # "to(dev)" equals "cuda()" if CUDA is available.
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

pretrained_name = "kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.4"
tokenizer = None
model = None

def generate(input, instruction="", model_name = "kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.4", min_length=100, max_length=300, top_k=50, top_p=0.95, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, temperature=1.9):
    global tokenizer, model
    # load model
    if (not tokenizer or not model):
        if model_name == "kobkrit/openthaigpt-0.1.0-alpha":
            config = PeftConfig.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, model_name).to(dev) # Use "to(dev)" instead of "cuda()" to make sure it works with cpu-only cases.
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>',unk_token='<|unk|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
            model = GPT2LMHeadModel.from_pretrained(model_name).to(dev) # Use "to(dev)" instead of "cuda()" to make sure it works with cpu-only cases.

    # inference
    if model_name == "kobkrit/openthaigpt-0.1.0-alpha":
        generated = tokenizer('<instruction>: ' + str(instruction) + ' <input>: ' + str(input) + ' <output>: ', max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(dev) # Use "to(dev)" instead of "cuda()" to make sure it works with cpu-only cases.
        generated = tokenizer("<|startoftext|>"+input, return_tensors="pt").input_ids.to(dev) # Use "to(dev)" instead of "cuda()" to make sure it works with cpu-only cases.

    with torch.no_grad():
        output = model.generate(input_ids=generated, top_k=top_k, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, 
            early_stopping=early_stopping, min_length=min_length, max_length=max_length, top_p=top_p, temperature=temperature)
        return tokenizer.decode(output[0], skip_special_tokens=True)

def zero(input, threshold=10):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=[input], model_id=pretrained_name)
    score = results['perplexities'][0]
    isGeneratedFromOpenThaiGPT = False
    if (score < threshold):
        isGeneratedFromOpenThaiGPT = True
    return {'perplexity':score, 'threshold':threshold, 'isGeneratedFromOpenThaiGPT':isGeneratedFromOpenThaiGPT}
