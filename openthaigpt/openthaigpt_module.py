"""Main module."""
import torch # Used to check if CUDA is available.
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Check if CUDA is available
if torch.cuda.is_available():
    # "to(dev)" equals "cuda()" if CUDA is available.
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

pretrained_name = "kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.3"

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_name, bos_token='<|startoftext|>',unk_token='<|unk|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained(pretrained_name).to(dev) # Use "to(dev)" instead of "cuda()" to make sure it works with cpu-only cases.

model.resize_token_embeddings(len(tokenizer))

def generate(input, max_length=300, top_k=50, top_p=0.95, num_beam=5, no_repeat_ngram_size=2, early_stopping=True, temperature=1.9):
    generated = tokenizer("<|startoftext|>"+input, return_tensors="pt").input_ids.to(dev) # Use "to(dev)" instead of "cuda()" to make sure it works with cpu-only cases.
    output = model.generate(generated, top_k=top_k, num_beams=num_beam, no_repeat_ngram_size=no_repeat_ngram_size, 
        early_stopping=early_stopping, max_length=max_length, top_p=top_p, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)