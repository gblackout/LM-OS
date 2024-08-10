import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from typing import Optional, List, Union
from utils import all_exists


def load_hf_model(
        base_model,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attn: bool = False,
        device_map='auto',
):
    assert not (load_in_4bit and load_in_8bit), 'both 4bit and 8bit are True, give me just one flag'

    add_kwargs = {'attn_implementation': "flash_attention_2"} if use_flash_attn else {}
    if any([load_in_8bit, load_in_4bit]):
        add_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        **add_kwargs
    )

    return model


def load_model_and_peft(
        base_model,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attn: bool = False,
        device_map='auto',
        # lora params, if you init one for training
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_dropout: Optional[float] = None,
        # lora path, if you load one for inference; or if a list of (path, adapter_name) given, i will load all of them
        # note the last lora is active by default
        peft_path: Union[str, List, None] = None
):

    model = load_hf_model(base_model, load_in_4bit, load_in_8bit, use_flash_attn, device_map)

    is_init_peft = all_exists(lora_r, lora_alpha, lora_target_modules, lora_dropout)
    is_load_peft = all_exists(peft_path)
    assert not (is_init_peft and is_load_peft), \
        'you give too many args, either give peft_path for me to load one, or the rest for me to init one'

    if is_init_peft:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
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

    elif is_load_peft:
        if isinstance(peft_path, str):
            model = PeftModel.from_pretrained(
                model,
                peft_path,
                torch_dtype=torch.float16
            )
            model.to('cuda')
        elif isinstance(peft_path, list):
            for ppath, name in peft_path:
                model = PeftModel.from_pretrained(
                    model,
                    ppath,
                    torch_dtype=torch.float16,
                    adapter_name=name
                )
            model.to('cuda')


    else:
        print('no lora params found; loaded based model')

    return model