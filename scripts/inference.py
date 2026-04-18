import gc
import pandas as pd
import torch
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
        label = full_text.split("### Response:\n")[-1].strip()
        return label

def evaluate_model(model_path, test_df):
    classifier = IntentClassification(model_path)
    correct = 0
    total = len(test_df)
    
    for index, row in test_df.iterrows():
        prediction = classifier(row['text'])
        if prediction == row['label']:
            correct += 1
            
    accuracy = (correct / total) * 100
    
    del classifier.model
    del classifier.tokenizer
    del classifier
    gc.collect()
    torch.cuda.empty_cache()
    
    return accuracy

if __name__ == "__main__":
    test_df = pd.read_csv("sample_data/test.csv")
    
    print("--------------------------------------------------")
    print("1. EVALUATING BASELINE MODEL (unsloth/Llama-3.1-8B)")
    print("--------------------------------------------------")
    baseline_accuracy = evaluate_model("unsloth/Llama-3.1-8B", test_df)
    
    print("\n--------------------------------------------------")
    print("2. EVALUATING FINE-TUNED MODEL (llama_lora)")
    print("--------------------------------------------------")
    finetuned_accuracy = evaluate_model("llama_lora", test_df)
    
    print("\n--------------------------------------------------")
    print("3. PERFORMANCE COMPARISON")
    print("--------------------------------------------------")
    print(f"Total Test Samples : {len(test_df)}")
    print(f"Baseline Accuracy  : {baseline_accuracy:.2f}%")
    print(f"Fine-tuned Accuracy: {finetuned_accuracy:.2f}%")
    print(f"Improvement        : {finetuned_accuracy - baseline_accuracy:.2f}%")
    print("--------------------------------------------------")