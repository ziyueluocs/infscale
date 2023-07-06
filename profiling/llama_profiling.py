import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from profile_in_json import get_model_inference_profile

def bert_input_constructor(batch_size, seq_len, tokenizer, batch_num=1):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * (batch_size * batch_num),
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * (batch_size * batch_num))
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs

with get_accelerator().device(0):
    tokenizer = LlamaTokenizer.from_pretrained('openlm-research/open_llama_7b')
    model = LlamaForCausalLM.from_pretrained('openlm-research/open_llama_7b', torch_dtype=torch.float16, device_map=0)
    
    prompt = 'Q: What is the largest animal?\nA:'
    input = dict(tokenizer(prompt, return_tensors="pt"))
    print(input)
    flops, macs, params = get_model_inference_profile(
        model,
        kwargs=input,
        print_profile=True,
        detailed=True,
        mode="generate",
        output_file="./llama_profile.json"
    )