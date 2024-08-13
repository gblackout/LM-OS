from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn
from transformers import Cache, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb
)
from utils import all_exists


class CompressorRetrieverConfig(PretrainedConfig):
    model_type = "CompressorRetriever"

    def __init__(self, base_model_name="CompressorRetriever", **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name


class CompressorRetriever(PreTrainedModel):
    config_class = CompressorRetrieverConfig
    COM_TOKEN = 0
    RET_TOKEN = 1
    TOKEN_SHIFT = 58888

    def __init__(self, config: CompressorRetrieverConfig, base_causallm: PreTrainedModel):
        super().__init__(config)
        self.base_causallm = base_causallm
        base_embedding = self.base_causallm.get_input_embeddings()
        self.mq_embedding = nn.Embedding(2, base_embedding.weight.shape[1])

        self.post_init()

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
    ):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        assert 'base_causallm' in kwargs, 'you need to specify base_causallm'
        model = cls(config, kwargs.pop('base_causallm'))

        # Load the state dict
        state_dict = torch.load(pretrained_model_name_or_path + "/pytorch_model.bin")

        # Only load the new embedding layer if it exists in the state dict
        if "mq_embedding.weight" in state_dict:
            model.mq_embedding.load_state_dict({
                "weight": state_dict["mq_embedding.weight"]
            })

        return model

    def save_pretrained(self, save_directory, **kwargs):
        # Save the config
        self.config.save_pretrained(save_directory)

        # Save only the new embedding layer
        state_dict = {
            "mq_embedding.weight": self.mq_embedding.state_dict()["weight"]
        }
        torch.save(state_dict, save_directory + "/pytorch_model.bin")

    def make_embedding(self, input_ids: torch.LongTensor):
        """
            given an input_ids mixed with base_causallm ids and mq ids (that are shifted by TOKEN_SHIFT), get
            the embeddings from the corresponding embedding table. This is done with torch.where masking.
        :param input_ids:
        :return:
        """
        emb1_mask = torch.tensor([input_id - self.TOKEN_SHIFT < 0 for input_id in input_ids]).to(input_ids.device)
        ind_emb1 = torch.where(emb1_mask, input_ids, torch.zeros_like(input_ids))
        ind_emb2 = torch.where(~emb1_mask, input_ids - self.TOKEN_SHIFT, torch.zeros_like(input_ids))
        emb_arr1 = self.base_causallm.get_input_embeddings()(ind_emb1)
        emb_arr2 = self.mq_embedding(ind_emb2)
        res = torch.where(emb1_mask.unsqueeze(-1), emb_arr1, emb_arr2)
        return res

    def forward(self, input_ids: torch.LongTensor = None, **kwargs):
        assert all_exists(input_ids), 'you have to give me input_ids'

        embedded = self.make_embedding(input_ids)
        return embedded

        # # Placeholder: pass the embedded input to the base model
        # # You might need to adjust this depending on your base model's forward method
        # outputs = self.base_model(inputs_embeds=embedded, **kwargs)
        #
        # return outputs
