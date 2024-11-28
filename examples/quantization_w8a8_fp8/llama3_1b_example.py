# from transformers import AutoTokenizer
# from llmcompressor.transformers import oneshot
# from llmcompressor.modifiers.quantization import QuantizationModifier
# from llmcompressor.transformers import SparseAutoModelForCausalLM

# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# model = SparseAutoModelForCausalLM.from_pretrained(
#   MODEL_ID, device_map="auto", torch_dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# # Configure the simple PTQ quantization
# recipe = QuantizationModifier(
#   targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# # Apply the quantization algorithm.
# oneshot(model=model, recipe=recipe)

# print(model)

# # # Save the model.
# SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
# model.save_pretrained(SAVE_DIR)
# tokenizer.save_pretrained(SAVE_DIR)


import time
from vllm import LLM
model = LLM("neuralmagic/Llama-3.2-1B-Instruct-FP8-dynamic")
# model = LLM("meta-llama/Llama-3.2-1B-Instruct")
st = time.time()
result = model.generate("Hello my name is")
et = time.time()
print(et - st)