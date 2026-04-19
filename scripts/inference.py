import gc
import os
import pandas as pd
import torch
import yaml
from unsloth import FastLanguageModel
import warnings
import transformers
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

class IntentClassification:
    def __init__(self, model_path):
        if isinstance(model_path, str) and model_path.endswith(".yaml"):
            with open(model_path, "r") as f:
                config = yaml.safe_load(f)
            model_name = config.get("model_path", "llama_lora")
            max_seq_length = config.get("max_seq_length", 2048)
            load_in_4bit = config.get("load_in_4bit", True)
        else:
            model_name = model_path
            max_seq_length = 2048
            load_in_4bit = True

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit
        )
        FastLanguageModel.for_inference(self.model)

        self.instruction = "Identify the banking intent for the following customer query."
        self.prompt_template = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n"
        )

    def __call__(self, message):
        input_text = self.prompt_template.format(self.instruction, message)
        inputs = self.tokenizer([input_text], return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=64, 
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in decoded_output:
            return decoded_output.split("### Response:")[-1].strip()
        return decoded_output.strip()

def evaluate_performance(model_path, test_df, label_text):
    classifier = IntentClassification(model_path)
    correct = 0
    total = len(test_df)
    
    print(f"--- Evaluating {label_text} ---")
    for index, row in test_df.iterrows():
        prediction = classifier(row['text'])
        if prediction.lower() == str(row['label']).lower():
            correct += 1
        
        if (index + 1) % 20 == 0:
            print(f"Progress: {index + 1}/{total}")
            
    accuracy = (correct / total) * 100
    
    return accuracy

if __name__ == "__main__":
    test_path = "sample_data/test.csv"
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        exit()
        
    test_df = pd.read_csv(test_path)
    
    baseline_model = "unsloth/Llama-3.1-8B"
    baseline_acc = evaluate_performance(baseline_model, test_df, "Baseline Model")
    
    finetuned_config = "configs/inference.yaml"
    finetuned_acc = evaluate_performance(finetuned_config, test_df, "Fine-tuned Model")
    
    print("\n" + "="*30)
    print("COMPARISON RESULTS (100 SAMPLES)")
    print(f"Baseline Accuracy:  {baseline_acc:.2f}%")
    print(f"Fine-tuned Accuracy: {finetuned_acc:.2f}%")
    print("="*30 + "\n")
    
    
    inference_tool = IntentClassification(finetuned_config)
    print("\n" + "="*30)
    print("--- USAGE EXAMPLE ---")
    custom_query = "My credit card was stolen, how can I block it immediately?"
    prediction = inference_tool(custom_query)
    
    print(f"Input: {custom_query}")
    print(f"Prediction: {prediction}")
    print("\n" + "="*30)