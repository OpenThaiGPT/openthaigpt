"""Main module."""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from evaluate import load
pretrained_name = "kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.4"

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_name, bos_token='<|startoftext|>',unk_token='<|unk|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained(pretrained_name).cuda()
model.resize_token_embeddings(len(tokenizer))

def generate(input, max_length=300, top_k=50, top_p=0.95, num_beam=5, no_repeat_ngram_size=2, early_stopping=True, temperature=1.9):
    generated = tokenizer("<|startoftext|>"+input, return_tensors="pt").input_ids.cuda()
    output = model.generate(generated, top_k=top_k, num_beams=num_beam, no_repeat_ngram_size=no_repeat_ngram_size, 
        early_stopping=early_stopping, max_length=max_length, top_p=top_p, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def zero(input, threshold=10):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=[input], model_id=pretrained_name)
    score = results['perplexities'][0]
    isGeneratedFromOpenThaiGPT = False
    if (score < threshold):
        isGeneratedFromOpenThaiGPT = True
    return {'perplexity':score, 'theshold':threshold, 'isGeneratedFromOpenThaiGPT':isGeneratedFromOpenThaiGPT}