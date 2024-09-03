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
from utils import any_exists, all_exists
from peft import PeftModel
import torch.nn.functional as F


class CompressorRetrieverConfig(PretrainedConfig):
    model_type = "CompressorRetriever"

    def __init__(self, base_model_name="CompressorRetriever", **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name


class CompressorRetriever(PreTrainedModel):
    config_class = CompressorRetrieverConfig
    COM_TOKEN = 0
    RET_TOKEN = 1
    TOKEN_SHIFT = 9_999_999

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(self, config: CompressorRetrieverConfig, base_causallm: PreTrainedModel):
        super().__init__(config)
        self.base_causallm = base_causallm

        # TODO maybe enumerate if can't find the one named `model`?
        model = getattr(base_causallm, 'model', None)
        if isinstance(model, PreTrainedModel):

            # TODO if base_causallm is PeftModel, then base_pretrainedlm is model_model need an elegant way for this
            model_model = getattr(model, 'model', None)
            if isinstance(model_model, PreTrainedModel):
                self.base_pretrainedlm = model_model
            else:
                self.base_pretrainedlm = model

        else:
            raise ValueError('cannot locate the base PreTrainedModel from the base_causallm obj')

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
            # TODO need to adpot the peft way to set the precision
            model.mq_embedding.weight.data = model.mq_embedding.weight.data.half()

        return model

    def save_pretrained(self, save_directory, **kwargs):

        # save lora
        # TODO i'm not sure if changing order will cause my stuff gets override
        if isinstance(self.base_causallm, PeftModel):
            self.base_causallm.save_pretrained(save_directory)

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
        # emb1_mask = torch.tensor([input_id - self.TOKEN_SHIFT < 0 for input_id in input_ids]).to(input_ids.device)
        emb1_mask = input_ids - self.TOKEN_SHIFT < 0
        ind_emb1 = torch.where(emb1_mask, input_ids, torch.zeros_like(input_ids))
        ind_emb2 = torch.where(~emb1_mask, input_ids - self.TOKEN_SHIFT, torch.zeros_like(input_ids))
        emb_arr1 = self.base_causallm.get_input_embeddings()(ind_emb1)
        # with torch.autocast(device_type="cuda"): # TODO adhoc, base embedding all casted to float32
        emb_arr2 = self.mq_embedding(ind_emb2)
        res = torch.where(emb1_mask.unsqueeze(-1), emb_arr1, emb_arr2)
        return res

    def forward(
            self,
            ctx_ids: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            compression_factor: Optional[float] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            is_generation: bool = False,
            is_demo_retrival: bool = False,
            ctx_attention_masks: Optional[List] = None,
            relevant_ctx_ids: Optional[torch.LongTensor] = None,
            **kwargs
    ):
        assert not all_exists(input_ids, inputs_embeds), 'you should only give either input_ids or inputs_embeds'

        if all_exists(input_ids):
            embedded = self.make_embedding(input_ids)
        elif all_exists(inputs_embeds):
            embedded = inputs_embeds
        else:
            raise ValueError(f'you should give either input_ids or inputs_embeds')

        if is_demo_retrival:

            input_state = self.compress(inputs_embeds=embedded, output_len=1) # TODO ad hoc state 1
            bsize = ctx_ids.shape[0]
            # build the mem tree
            full_mem_trees = []
            n_ctx = ctx_ids.shape[1]
            for ctx_ind in range(n_ctx): # TODO no batch here
                seg_mem_tree = []
                ctx_id = ctx_ids[:, ctx_ind, :]
                cur_seg = self.base_causallm.get_input_embeddings()(ctx_id)
                # TODO sth wrong with compression_factor, can't get element of it nor do anything
                for factor in [1, 'single']:
                    if factor == 'single':
                        output_len = 1
                        cur_seg = torch.cat([input_state, cur_seg], dim=1)  # (bsize, len, dim)
                        com_embeds = self.compress(inputs_embeds=cur_seg,output_len=output_len,)

                    else:
                        output_len = 50 # TODO instead of int(cur_seg.shape[1] / factor), we use 50 to cut down cost
                        cur_seg = torch.cat([input_state, cur_seg], dim=1)  # (bsize, len, dim)
                        ctx_attention_mask = torch.cat(
                            [
                                torch.full(input_state.shape[:2], 1).to(input_state.device),
                                ctx_attention_masks[:, ctx_ind, :],
                                torch.full((bsize, output_len), 1).to(input_state.device),
                            ],
                            dim=1
                        )
                        com_embeds = self.compress(
                            inputs_embeds=cur_seg,
                            output_len=output_len,
                            attention_mask=ctx_attention_mask
                        )

                    seg_mem_tree.append(com_embeds)
                    cur_seg = com_embeds

                full_mem_trees.append(seg_mem_tree[::-1])

            # gather top level and get attn
            top_mem = torch.cat([mem_tree[0] for mem_tree in full_mem_trees], dim=1) # (bsize, len, dim)
            # (bsize, n_ctx, seg_len, dim)
            main_mem = torch.cat([mem_tree[1][:, None , :, :] for mem_tree in full_mem_trees], dim=1)
            bsize, n_ctx, seg_len, model_dim = main_mem.shape
            q_len = 2
            top_k = 2
            mem_len = top_mem.shape[1]
            query_embeds = self.map_forward(inputs_embeds=embedded, output_len=q_len, output_token_id=self.RET_TOKEN)

            outputs = self.base_pretrainedlm(
                inputs_embeds=torch.cat([top_mem, query_embeds], dim=1),
                output_attentions=True,
                return_dict=True,
            )
            assert isinstance(outputs, BaseModelOutputWithPast)
            attn = outputs.attentions[-1] # (bsize, n_head, mem_len+q_len, mem_len+q_len)
            # TODO i'm not sure if averaging is good
            attn = attn.mean(dim=1)

            # Get the attention scores for the last m rows and the first n columns
            attn_last_m_first_n = attn[:, mem_len:, :mem_len]  # (b, q_len, mem_len)

            # Get the top-k indices and values for each of the m rows
            topk_vals, topk_indices = torch.topk(attn_last_m_first_n, top_k, dim=-1)
            topk_softmax = F.softmax(topk_vals, dim=1) # (bsize, q_len, top_k)

            # (bsize, q_len, top_k, seg_len, dim)
            ind_expanded = topk_indices[..., None, None].expand(-1, -1, -1, seg_len, model_dim)
            # (bsize, q_len, n_ctx, seg_len, dim)
            x_expanded = main_mem[:, None, ...].expand(-1, q_len, -1, -1, -1)
            # (bsize, q_len, top_k, seg_len, dim)
            topk_segs = torch.gather(x_expanded, 2, ind_expanded)

            # Compute the weighted sum (bsize, q_len, seg_len, dim)
            retrieved_mem = torch.sum(topk_segs * topk_softmax[..., None, None], dim=2)
            # TODO (bsize, q_len * seg_len, dim), this exceeds the q_len quota
            retrieved_mem = retrieved_mem.view(bsize, -1, model_dim)

            # final embedding
            embedded = torch.cat((retrieved_mem, embedded), dim=1)  # (b, s_len, dim) cat at s_len

            # del mem tree and intermediate outputs
            del outputs, full_mem_trees, attn, top_mem, main_mem, query_embeds, attn_last_m_first_n

            # TODO assuming labels/attn_masks are same shape as inputs
            if all_exists(labels):
                ret_dummy_labels = torch.full(retrieved_mem.shape[:2], -100, dtype=torch.long, device=labels.device)
                labels = torch.cat((ret_dummy_labels, labels), dim=-1)  # (bsize, dummy_len + actual_len)

            if all_exists(attention_mask):
                ret_masks = torch.full(retrieved_mem.shape[:2], 1, dtype=torch.long, device=attention_mask.device)
                attention_mask = torch.cat((ret_masks, attention_mask), dim=-1)  # (bsize, dummy_len + actual_len)

        if (not is_demo_retrival) and all_exists(ctx_ids, compression_factor):
            output_len = int(ctx_ids.shape[1] / 1) # TODO sth wrong with compression_factor, can't get element of it nor do anything
            com_embeds = self.compress(input_ids=ctx_ids, output_len=output_len)
            embedded = torch.cat((com_embeds, embedded), dim=1)  # (b, s_len, dim) cat at s_len

            # TODO assuming labels/attn_masks are same shape as inputs
            if all_exists(labels):
                com_dummy_labels = torch.full(com_embeds.shape[:2], -100, dtype=torch.long, device=labels.device)
                labels = torch.cat((com_dummy_labels, labels), dim=-1) # (bsize, dummy_len + actual_len)

            if all_exists(attention_mask):
                com_masks = torch.full(com_embeds.shape[:2], 1, dtype=torch.long, device=attention_mask.device)
                attention_mask = torch.cat((com_masks, attention_mask), dim=-1)  # (bsize, dummy_len + actual_len)

        kwargs['inputs_embeds'] = embedded
        kwargs['attention_mask'] = attention_mask

        if all_exists(labels): # TODO this is ugly, need to have two forward maybe?
            kwargs['labels'] = labels
            outputs = self.base_causallm(**kwargs)
        elif is_generation: # TODO adhoc during generation we simply use base causal
            outputs = self.base_causallm(**kwargs)
        else:
            outputs = self.base_pretrainedlm(**kwargs)

        return outputs

    def compress(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_len: Optional[int] = None,
            **forward_kwargs
    ):
        assert not all_exists(input_ids, inputs_embeds), 'you should only give either input_ids or inputs_embeds'

        if all_exists(input_ids):
            bsize = input_ids.shape[0]
            output_tokens = torch.full(
                (bsize, output_len),
                self.COM_TOKEN + self.TOKEN_SHIFT,
                dtype=torch.long
            ).to(input_ids.device)
            concated_inputs = torch.cat((input_ids, output_tokens), dim=-1)
            outputs = self(input_ids=concated_inputs, return_dict=True, **forward_kwargs)

        elif all_exists(inputs_embeds):
            bsize = inputs_embeds.shape[0]
            output_tokens = torch.full((bsize, output_len), self.COM_TOKEN, dtype=torch.long).to(inputs_embeds.device)
            output_embeds = self.mq_embedding(output_tokens)
            concated_inputs = torch.cat((inputs_embeds, output_embeds), dim=1) # (b, s_len, dim) cat at s_len
            outputs = self(inputs_embeds=concated_inputs, return_dict=True, **forward_kwargs)
        else:
            raise ValueError(f'you should give either input_ids or inputs_embeds')

        # TODO attention mask to be implemented


        assert isinstance(outputs, BaseModelOutputWithPast)

        hidden_states = outputs.last_hidden_state
        output_hidden_states = hidden_states[:, -output_len:, :]

        return output_hidden_states

    def map_forward(
            self,
            output_len: int,
            output_token_id: int,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        assert not all_exists(input_ids, inputs_embeds), 'you should only give either input_ids or inputs_embeds'
        assert output_token_id in (self.COM_TOKEN, self.RET_TOKEN), f'unknown output_token_id {output_token_id}'

        if all_exists(input_ids):
            bsize = input_ids.shape[0]
            output_tokens = torch.full(
                (bsize, output_len),
                output_token_id + self.TOKEN_SHIFT,
                dtype=torch.long
            ).to(input_ids.device)
            concated_inputs = torch.cat((input_ids, output_tokens), dim=-1)
            outputs = self(input_ids=concated_inputs, return_dict=True)

        elif all_exists(inputs_embeds):
            bsize = inputs_embeds.shape[0]
            output_tokens = torch.full((bsize, output_len), output_token_id, dtype=torch.long).to(inputs_embeds.device)
            output_embeds = self.mq_embedding(output_tokens)
            concated_inputs = torch.cat((inputs_embeds, output_embeds), dim=1) # (b, s_len, dim) cat at s_len
            outputs = self(inputs_embeds=concated_inputs, return_dict=True)

        else:
            raise ValueError(f'you should give either input_ids or inputs_embeds')

        # TODO attention mask to be implemented

        assert isinstance(outputs, BaseModelOutputWithPast)

        hidden_states = outputs.last_hidden_state
        output_hidden_states = hidden_states[:, -output_len:, :]

        return output_hidden_states

    def demo_compress_generation(
            self,
            ctx_ids: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            compression_factor: Optional[float] = None,
            **kwargs
    ):

        output_len = ctx_ids.shape[1] // compression_factor
        com_embeds = self.compress(input_ids=ctx_ids, output_len=output_len)

        # NOTE using bas_causallm's embedding table
        input_embeds = self.base_causallm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat((com_embeds, input_embeds), dim=1) # (b, s_len, dim) cat at s_len

        return self.base_causallm.generate(inputs_embeds=inputs_embeds, **kwargs)

    def demo_retrieval_generation(
            self,
            ctx_ids: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            ctx_attention_masks: Optional[List] = None,
            relevant_ctx_ids: Optional[torch.LongTensor] = None,
            **kwargs
    ):

        embedded = self.make_embedding(input_ids)

        input_state = self.compress(inputs_embeds=embedded, output_len=1)  # TODO ad hoc state 1

        bsize = ctx_ids.shape[0]
        # build the mem tree
        full_mem_trees = []
        n_ctx = ctx_ids.shape[1]
        for ctx_ind in range(n_ctx):  # TODO no batch here
            seg_mem_tree = []
            ctx_id = ctx_ids[:, ctx_ind, :]
            cur_seg = self.base_causallm.get_input_embeddings()(ctx_id)
            # TODO sth wrong with compression_factor, can't get element of it nor do anything
            for factor in [1, 'single']:
                if factor == 'single':
                    output_len = 1
                    cur_seg = torch.cat([input_state, cur_seg], dim=1)  # (bsize, len, dim)
                    com_embeds = self.compress(inputs_embeds=cur_seg, output_len=output_len, )

                else:
                    output_len = 50  # TODO instead of int(cur_seg.shape[1] / factor), we use 50 to cut down cost
                    cur_seg = torch.cat([input_state, cur_seg], dim=1)  # (bsize, len, dim)
                    ctx_attention_mask = torch.cat(
                        [
                            torch.full(input_state.shape[:2], 1).to(input_state.device),
                            ctx_attention_masks[:, ctx_ind, :],
                            torch.full((bsize, output_len), 1).to(input_state.device),
                        ],
                        dim=1
                    )
                    com_embeds = self.compress(
                        inputs_embeds=cur_seg,
                        output_len=output_len,
                        attention_mask=ctx_attention_mask
                    )

                seg_mem_tree.append(com_embeds)
                cur_seg = com_embeds

            full_mem_trees.append(seg_mem_tree[::-1])

        # gather top level and get attn
        top_mem = torch.cat([mem_tree[0] for mem_tree in full_mem_trees], dim=1)  # (bsize, len, dim)
        # (bsize, n_ctx, seg_len, dim)
        main_mem = torch.cat([mem_tree[1][:, None, :, :] for mem_tree in full_mem_trees], dim=1)
        bsize, n_ctx, seg_len, model_dim = main_mem.shape
        q_len = 2
        top_k = 2
        mem_len = top_mem.shape[1]
        query_embeds = self.map_forward(inputs_embeds=embedded, output_len=q_len, output_token_id=self.RET_TOKEN)

        outputs = self.base_pretrainedlm(
            inputs_embeds=torch.cat([top_mem, query_embeds], dim=1),
            output_attentions=True,
            return_dict=True,
        )
        assert isinstance(outputs, BaseModelOutputWithPast)
        attn = outputs.attentions[-1]  # (bsize, n_head, mem_len+q_len, mem_len+q_len)
        # TODO i'm not sure if averaging is good
        attn = attn.mean(dim=1)

        # Get the attention scores for the last m rows and the first n columns
        attn_last_m_first_n = attn[:, mem_len:, :mem_len]  # (b, q_len, mem_len)

        # Get the top-k indices and values for each of the m rows
        topk_vals, topk_indices = torch.topk(attn_last_m_first_n, top_k, dim=-1)
        topk_softmax = F.softmax(topk_vals, dim=1)  # (bsize, q_len, top_k)

        # (bsize, q_len, top_k, seg_len, dim)
        ind_expanded = topk_indices[..., None, None].expand(-1, -1, -1, seg_len, model_dim)
        # (bsize, q_len, n_ctx, seg_len, dim)
        x_expanded = main_mem[:, None, ...].expand(-1, q_len, -1, -1, -1)
        # (bsize, q_len, top_k, seg_len, dim)
        topk_segs = torch.gather(x_expanded, 2, ind_expanded)

        # Compute the weighted sum (bsize, q_len, seg_len, dim)
        retrieved_mem = torch.sum(topk_segs * topk_softmax[..., None, None], dim=2)
        # TODO (bsize, q_len * seg_len, dim), this exceeds the q_len quota
        retrieved_mem = retrieved_mem.view(bsize, -1, model_dim)

        embedded = torch.cat((retrieved_mem, embedded), dim=1)  # (b, s_len, dim) cat at s_len

        # del mem tree and intermediate outputs
        del outputs, full_mem_trees, attn, top_mem, main_mem, query_embeds, attn_last_m_first_n
        del kwargs['attention_mask']
        return self.base_causallm.generate(inputs_embeds=embedded, **kwargs), topk_indices

    def prepare_inputs_for_generation(
            self,
            *args,
            **kwargs,
    ):
        """
            implement this func, so that we can call cr_model.generate()
        :param args:
        :param kwargs:
        :return:
        """
        return self.base_causallm.prepare_inputs_for_generation(*args, **kwargs)
