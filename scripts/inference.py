import yaml
from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

class IntentClassification:
    def __init__(self, model_path):
        # Doc file cau hinh mac dinh tu config/inference.yaml neu can
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message):
        instruction = "Identify the banking intent for the following customer query."
        input_text = PROMPT_TEMPLATE.format(instruction, message, "")
        
        inputs = self.tokenizer([input_text], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=64)
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Tach lay phan phan hoi sau Response
        label = full_text.split("### Response:\n")[-1].strip()
        return label

if __name__ == "__main__":
    # Su dung duong dan den checkpoint da luu
    classifier = IntentClassification("llama_lora")
    msg = "I just had my card get declined at the ATM!"
    print(f"Query: {msg}")
    print(f"Intent: {classifier(msg)}")