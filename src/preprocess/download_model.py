from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# tokenizer.save_pretrained('./models/Llama-3.1-8B')
# model.save_pretrained('./models/Llama-3.1-8B')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

tokenizer.save_pretrained('./models/Qwen3-8B')
model.save_pretrained('./models/Qwen3-8B')
