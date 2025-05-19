from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    deepspeed,
)

model_path = "/home/jovyan/lxq/model_dir/Qwen/Qwen2___5-0___5B-Instruct"
# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(model_path)
# 加载 LoRA 参数
lora_model = "/home/jovyan/lxq/finetune/qwen_fintune/output_qwen2/checkpoint-2346"
model = PeftModel.from_pretrained(base_model, lora_model)

test_texts = {
    'instruction': "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
    'input': "文本:航空动力学报JOURNAL OF AEROSPACE POWER1998年 第4期 No.4 1998科技期刊管路系统敷设的并行工程模型研究*陈志英*　*　马　枚北京航空航天大学【摘要】　提出了一种应用于并行工程模型转换研究的标号法，该法是将现行串行设计过程(As-is)转换为并行设计过程(To-be)。本文应用该法将发动机外部管路系统敷设过程模型进行了串并行转换，应用并行工程过程重构的手段，得到了管路敷设并行过程模型。"
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# 推理
input_text = "小儿肥胖超重该如何治疗"
#inputs = tokenizer(input_text, return_tensors="pt")
device = "cuda"
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#model_inputs = tokenizer([text], return_tensors="pt")
model_inputs = tokenizer(input_text, return_tensors="pt")
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)