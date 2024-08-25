import json
import os
from typing import List, Optional
import transformers
from datasets import load_dataset
import logging
from utils import all_exists
from utils.misc import make_output_dir
from generate import MyLlamaTokenizer
import fire
import wandb
import numpy as np
from model.model_utils import load_hf_model
from model.compressor_retriever import CompressorRetriever
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


def prepare_dataset(data_path, val_data_path, tokenizerlm, val_size):

    with open(data_path, 'r') as f:
        data = json.load(f)
        train_data = [e for e in data if e['src'] == 'train']
        val_data = [e for e in data if e['src'] == 'test']

    # if all_exists(val_data_path):
    #     with open(val_data_path, 'r') as f:
    #         val_data = json.load(f)
    # else:
    #     np.random.shuffle(train_data)
    #     val_data, train_data = train_data[:val_size], train_data[val_size:]

    # add required entries and save the processed datasets
    processed_train_fp, processed_valid_fp = 'data/train_data.json', 'data/valid_data.json'
    for save_fp, data in [[processed_valid_fp, val_data], [processed_train_fp, train_data]]:
        for sample in data:
            if 'orig_data' in sample:
                del sample['orig_data']

        logging.info(f'{len(data)} data saved in {save_fp}')

        with open(save_fp, 'w') as f:
            json.dump(data, f)

    def tokenize(sample):
        main_dict = tokenizerlm(
            [
                [sample['input'], 'False'],
                [sample['label'], 'True'],
            ],
            add_bos=False,
            add_eos=False,
            single_dim=True,
            my_token2llama_token=True
        )
        main_dict['ctx_ids'] = tokenizerlm(
            sample['ctx'],
            add_bos=False,
            add_eos=False,
            single_dim=True,
            my_token2llama_token=True
        )['input_ids']
        main_dict['compression_factor'] = 1

        return main_dict

    data_files = {'train': processed_train_fp, 'test': processed_valid_fp}
    data = load_dataset("json", data_files=data_files)

    train_data = data['train'].shuffle().map(tokenize)
    val_data = data['test'].shuffle().map(tokenize)

    remove_cols = ['input', 'label', 'src', 'ctx']
    train_data = train_data.remove_columns(remove_cols)
    val_data = val_data.remove_columns(remove_cols)

    return train_data, val_data


def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    load_in_8bit: bool = True,
    val_data_path: Optional[str] = None,
    val_size: int = 3000,
    output_dir: Optional[str] = None,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 200,
    save_steps: int = 200,
    save_total_limit: int = 30,
    # tokenizer setting
    max_length: int = 2048,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    device_map: str = "auto",
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    use_wandb: bool = True,
    wandb_project: str = "naive_translate_llama_sft",
    wandb_run_name: str = "default_run",
    # resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    assert isinstance(lora_target_modules, list)

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
        )

    if not os.path.isdir(base_model):
        print('base_model does not seem to be a file path, will try to load it with from_pretrained anyway')
    assert os.path.isfile(data_path), 'cannot locate data file'
    real_output_dir = make_output_dir(output_dir)

    tokenizerlm = MyLlamaTokenizer(base_model)
    tokenizerlm.llama_tokenizer.padding_side = "left"  # Allow batched inference

    # prep data
    train_data, val_data = prepare_dataset(data_path, val_data_path, tokenizerlm, val_size)

    model = load_hf_model(base_model, load_in_8bit=True)
    # freeze base model weights and make embedding requires_grad
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )

    # apply lora
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.to('cuda')
    model.print_trainable_parameters()

    cr_model = CompressorRetriever.from_pretrained('logs/cr_test_save', base_causallm=model)
    cr_model.to('cuda')
    cr_model.mq_embedding.weight.data = cr_model.mq_embedding.weight.data.float() # TODO adhoc

    for name, param in cr_model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    trainer = transformers.Trainer(
        model=cr_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=1, # TODO adhoc
            eval_accumulation_steps=32, # TODO adhoc
            gradient_accumulation_steps=batch_size // micro_batch_size,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=real_output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if all_exists(val_data_path) else False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else 'none',
            run_name=wandb_run_name if use_wandb else None,
            remove_unused_columns=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False}
        ),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizerlm.llama_tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
    )

    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(real_output_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
