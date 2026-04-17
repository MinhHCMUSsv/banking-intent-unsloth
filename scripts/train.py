import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def load_config(path="configs/train.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def formatting_func(examples, tokenizer):
    texts = []
    instruction = "Identify the banking intent for the following customer query."
    for text, label in zip(examples["text"], examples["label"]):
        formatted = PROMPT_TEMPLATE.format(instruction, text, label) + tokenizer.eos_token
        texts.append(formatted)
    return {"text": texts}

def main():
    cfg = load_config()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["model"]["load_in_4bit"]
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"]
    )
    
    train_ds = Dataset.from_pandas(pd.read_csv("sample_data/train.csv"))
    val_ds = Dataset.from_pandas(pd.read_csv("sample_data/val.csv"))
    
    train_ds = train_ds.map(lambda x: formatting_func(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: formatting_func(x, tokenizer), batched=True)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        args=SFTConfig(
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            warmup_steps=cfg["training"]["warmup_steps"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            eval_strategy=cfg["training"]["eval_strategy"],
            eval_steps=cfg["training"]["eval_steps"],
            logging_steps=cfg["training"]["logging_steps"],
            optim=cfg["training"]["optim"],
            weight_decay=cfg["training"]["weight_decay"],
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
            seed=cfg["training"]["seed"],
            output_dir=cfg["training"]["output_dir"],
            report_to="none"
        )
    )
    
    trainer.train()
    model.save_pretrained(cfg["training"]["save_model_path"])
    tokenizer.save_pretrained(cfg["training"]["save_model_path"])

if __name__ == "__main__":
    main()