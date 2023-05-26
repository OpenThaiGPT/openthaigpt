"""Main module."""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from transformers.models.mt5 import MT5Tokenizer
from transformers import DataCollatorForLanguageModeling
from peft import PeftModel, PeftConfig
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from evaluate import load
import torch
import sys

pretrained_name = "kobkrit/openthaigpt-0.1.0-beta"
tokenizer = None
model = None

def generate(input, instruction="", model_name = "kobkrit/openthaigpt-0.1.0-beta", min_length=40, max_length=256, top_k=40, top_p=0.75, num_beams=1, no_repeat_ngram_size=0, early_stopping=True, temperature=0.1, load_8bit=False):
    global tokenizer, model
    # load model
    if (not tokenizer or not model):
        if model_name == "kobkrit/openthaigpt-0.1.0-beta":
            base_model = 'decapoda-research/llama-7b-hf'
            lora_weights = 'kobkrit/openthaigpt-0.1.0-beta'

            tokenizer = LlamaTokenizer.from_pretrained(base_model)
            model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )

            # unwind broken decapoda-research config
            model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

            if not load_8bit:
                model.half()  # seems to fix bugs for some users.

            # model.eval()
            if torch.__version__ >= "2" and sys.platform != "win32":
                model = torch.compile(model)

        elif model_name == "kobkrit/openthaigpt-0.1.0-alpha":
            config = PeftConfig.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, model_name).cuda()
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>',unk_token='<|unk|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
            model = GPT2LMHeadModel.from_pretrained(model_name).cuda()

    # inference
    if model_name == "kobkrit/openthaigpt-0.1.0-beta":
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            early_stopping=early_stopping,
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams
        )
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s).split("### Response:")[1].strip()
        return output

    elif model_name == "kobkrit/openthaigpt-0.1.0-alpha":
        generated = tokenizer('<instruction>: ' + str(instruction) + ' <input>: ' + str(input) + ' <output>: ', max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            output = model.generate(input_ids=generated, top_k=top_k, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, 
                early_stopping=early_stopping, min_length=min_length, max_length=max_length, top_p=top_p, temperature=temperature)
            return tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        generated = tokenizer("<|startoftext|>"+input, return_tensors="pt").input_ids.cuda()
        
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