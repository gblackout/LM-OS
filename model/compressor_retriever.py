from transformers import LlamaForCausalLM

def foo(model: LlamaForCausalLM):
    model.resize_token_embeddings()


