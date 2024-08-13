from generate import simple_generate_with_tokenization, MyLlamaTokenizer
from transformers import AutoTokenizer, AutoModel
from model.model_utils import load_hf_model
from model.compressor_retriever import CompressorRetriever, CompressorRetrieverConfig
import torch


def main():
    model = load_hf_model('storage/models/hugging_face_models/Meta-Llama-3-8B', load_in_8bit=True)
    tokenizer = MyLlamaTokenizer('storage/models/hugging_face_models/Meta-Llama-3-8B')
    #
    # out = simple_generate_with_tokenization(
    #     model,
    #     input_str='hello 1 2 3 4',
    #     tokenizer=tokenizer,
    #     max_new_tokens=10,
    # )
    #
    # print(out)

    # config = CompressorRetrieverConfig()
    # cr_model = CompressorRetriever(config, model)
    # cr_model.save_pretrained('logs/cr_test_save')

    cr_model = CompressorRetriever.from_pretrained('logs/cr_test_save', base_causallm=model)
    cr_model.to('cuda')
    assert isinstance(cr_model, CompressorRetriever)

    tokens = tokenizer('this is a test message', return_pt=False)
    input_ids = tokens['input_ids']
    input_ids += [cr_model.RET_TOKEN + cr_model.TOKEN_SHIFT for _ in range(10)]
    input_ids = torch.tensor(input_ids, dtype=torch.int64).to('cuda')

    embedd = cr_model(input_ids)
    print(embedd)





if __name__ == '__main__':
    main()