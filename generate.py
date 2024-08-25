from transformers import GenerationConfig, PreTrainedModel, LlamaTokenizer, AutoTokenizer
from utils import all_exists
from typing import Union, List, Optional, Iterable
import torch
from dataclasses import dataclass


class MyLlamaTokenizer:
    """
        L2 and L3 use different configs and somehow the tokenizer behaviors are different too. I make this class
        to wrap these two and provide a unified behavior
    """

    # insert this into your text, and have my_token2llama_token on to replace the bos/eos into corresponding L2/L3 ones
    # this is needed for traj sft, because I need to insert it for every Response and Observation
    MY_BOS_TOKEN = '<!BOS!>'
    MY_EOS_TOKEN = '<!EOS!>'

    def __init__(
            self,
            llama_tokenizer: Union[AutoTokenizer, str],
            set_pad_token_to_eos_token: bool = True
    ):
        if isinstance(llama_tokenizer, str):
            llama_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer)
        self.llama_tokenizer = llama_tokenizer
        # make sure the bos/eos tokens are there
        assert all_exists(self.llama_tokenizer.bos_token, self.llama_tokenizer.eos_token)
        # make sure tokenizer not adding bos and eos
        self.llama_tokenizer.add_bos_token = False
        self.llama_tokenizer.add_eos_token = False
        # both L2 and L3 does not define pad_token
        if set_pad_token_to_eos_token:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(
            self,
            seq: Union[str, List[Iterable]],
            add_bos: bool = False,
            add_eos: bool = False,
            single_dim: bool = False,
            return_pt: bool = True,
            my_token2llama_token: bool = False
    ):
        """
        :param seq:
            seq should be either:
                1. a str
                2. a list of (s, trainable_flag) like [(s1, f1), (s2, f2), ...], where trainable_flag is a bool or
                a str 'True' or 'False' (had this alternative because of the stupid load_datasets) indicating if
                this seq is trainable or not
        :param my_token2llama_token:
            when set to True, I will replace those bos/eos strs in text into llama ones
        :return:
        """
        seq_ls = [[seq, True]] if isinstance(seq, str) else seq

        input_ids, att_mask, labels = [], [], []
        for seq, trainable_flag in seq_ls:

            if my_token2llama_token:
                seq = seq.replace(self.MY_BOS_TOKEN, self.llama_tokenizer.bos_token)
                seq = seq.replace(self.MY_EOS_TOKEN, self.llama_tokenizer.eos_token)

            input_res = self.llama_tokenizer(
                seq,
                truncation=False,
                padding=False,
                add_special_tokens=False
            )
            input_ids.extend(input_res['input_ids'])
            att_mask.extend(input_res['attention_mask'])
            trainable_flag = trainable_flag if isinstance(trainable_flag, bool) else trainable_flag == 'True'
            labels.extend([e if trainable_flag else -100 for e in input_res['input_ids']])

        # manually adding bos/eos tokens
        l, lm = [self.llama_tokenizer.bos_token_id] if add_bos else [], [1] if add_bos else []
        r, rm = [self.llama_tokenizer.eos_token_id] if add_eos else [], [1] if add_bos else []
        input_ids = l + input_ids + r
        att_mask = lm + att_mask + rm
        labels = l + labels + r # not setting labels of bos/eos to -100 should be fine

        res = {
            'input_ids': input_ids,
            'attention_mask': att_mask,
            'labels': labels
        }

        if return_pt:
            for k, v in res.items():
                res[k] = torch.tensor(v if single_dim else [v], dtype=torch.int64)

        return res

    def decode(self, seq, skip_special_tokens=False):
        return self.llama_tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)


def simple_generate(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        stopplist: Optional[list] = None
):
    # take care the input_id shape
    # if dim=1, then add another one
    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]
    assert (input_ids.shape[0] == 1) and (len(input_ids.shape) == 2)

    model_is_training = model.training
    model.eval()
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids.to('cuda'),
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopplist
        )

    # restore the original state
    if model_is_training:
        model.train()

    return generation_output[0]


@dataclass
class GenerationOutputWithTokenization:
    full_str: Optional[str] = None
    input_str: Optional[str] = None
    resp_str: Optional[str] = None
    full_ids: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    resp_ids: Optional[torch.Tensor] = None

    def to_list(self):
        return [
            self.full_str, self.input_str, self.resp_str,
            self.full_ids, self.input_ids, self.resp_ids
        ]

    def to_dict(self):
        return {
            'full_str': self.full_str, 'input_str': self.input_str, 'resp_str': self.resp_str,
            'full_ids': self.full_ids, 'input_ids': self.input_ids, 'resp_ids': self.resp_ids
        }


def simple_generate_with_tokenization(
    model: PreTrainedModel,
    input_str: str,
    tokenizer: Union[MyLlamaTokenizer, LlamaTokenizer],
    max_new_tokens: int,
    generation_config: Optional[GenerationConfig] = None,
    skip_special_tokens: bool = True,
    **tokenizer_kwargs
):

    inputs = tokenizer(input_str, **tokenizer_kwargs)
    input_ids = inputs['input_ids'].to('cuda')
    input_id_len = len(input_ids[0])

    full_ids = simple_generate(
        model,
        input_ids,
        max_new_tokens,
        generation_config
    )
    resp_ids = full_ids[input_id_len:]

    full_str = tokenizer.decode(full_ids, skip_special_tokens=skip_special_tokens)
    resp_str = tokenizer.decode(resp_ids, skip_special_tokens=skip_special_tokens)

    return GenerationOutputWithTokenization(
        full_str=full_str, input_str=input_str, resp_str=resp_str,
        full_ids=full_ids, input_ids=input_ids, resp_ids=resp_ids
    )
