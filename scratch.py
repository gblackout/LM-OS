from generate import simple_generate_with_tokenization, MyLlamaTokenizer, simple_generate
from transformers import AutoTokenizer, AutoModel
from model.model_utils import load_hf_model
from model.compressor_retriever import CompressorRetriever, CompressorRetrieverConfig
import torch
from copy import deepcopy
from utils.misc import jload, jdump, PCYAN_L, PCYAN_R, PGREEN_L, PGREEN_R
from tqdm import tqdm


def main():
    model = load_hf_model('storage/models/hugging_face_models/Meta-Llama-3-8B', load_in_8bit=True)
    tokenizer = MyLlamaTokenizer('storage/models/hugging_face_models/Meta-Llama-3-8B')

    compress_test = jload('data/compress_test_folio_300.json')

    for sample in tqdm(compress_test):
        s = f'{sample["ctx"]}\n\n{sample["input"]}'
        print(f'{PCYAN_L}{sample["label"]}{PCYAN_R}')
        inputs = tokenizer(s, my_token2llama_token=True)
        input_ids = inputs['input_ids'].to('cuda')
        input_id_len = len(input_ids[0])

        full_ids = simple_generate(
            model,
            input_ids,
            max_new_tokens=10,
            generation_config=None
        )
        resp_ids = full_ids[input_id_len:]
        print(f'{PGREEN_L}{tokenizer.decode(resp_ids, skip_special_tokens=True)}{PGREEN_R}')

        sample['2shot-pred'] = tokenizer.decode(resp_ids, skip_special_tokens=True)
    jdump(compress_test, 'data/compress_test_folio_300_2shot-pred.json')

    # sample = compress_test[0]
    # s = f'{sample["ctx"]}\n\n{sample["input"]}'
    # print(f'GT: {sample["label"]}')
    #
    # inputs = tokenizer(s, my_token2llama_token=True)
    # input_ids = inputs['input_ids'].to('cuda')
    # input_id_len = len(input_ids[0])
    #
    # full_ids = simple_generate(
    #     model,
    #     input_ids,
    #     max_new_tokens=50,
    #     generation_config=None
    # )
    # resp_ids = full_ids[input_id_len:]
    # print(tokenizer.decode(resp_ids, skip_special_tokens=True))

    # out = simple_generate_with_tokenization(
    #     model,
    #     input_str=s,
    #     tokenizer=tokenizer,
    #     max_new_tokens=50,
    #     my_token2llama_token=True
    # )
    # print(out.resp_str)

    # # config = CompressorRetrieverConfig()
    # # cr_model = CompressorRetriever(config, model)
    # # cr_model.save_pretrained('logs/cr_test_save')
    #
    # cr_model = CompressorRetriever.from_pretrained('logs/cr_test_save', base_causallm=model)
    # cr_model.to('cuda')
    # assert isinstance(cr_model, CompressorRetriever)
    #
    # ctx_tokens = tokenizer('test 1 2 3 4', return_pt=True)
    # tokens = tokenizer('test', return_pt=True)
    #
    # # input_ids = tokens['input_ids']
    # # input_ids += [cr_model.RET_TOKEN + cr_model.TOKEN_SHIFT for _ in range(10)]
    # # input_ids = torch.tensor([input_ids, deepcopy(input_ids)], dtype=torch.int64).to('cuda')
    #
    # cr_model.eval()
    # with torch.no_grad():
    #     generation_output = cr_model.base_causallm.generate(
    #         input_ids=tokens['input_ids'].to('cuda'),
    #         generation_config=None,
    #         max_new_tokens=5,
    #     )
    #
    # print(generation_output[0])
    # print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
    # print('===')
    #
    # with torch.no_grad():
    #     generation_output = cr_model.demo_compress_generation(
    #         ctx_ids=ctx_tokens['input_ids'].to('cuda'),
    #         input_ids=tokens['input_ids'].to('cuda'),
    #         compression_factor=1,
    #         generation_config=None,
    #         max_new_tokens=5,
    #     )
    #
    # print(generation_output[0])
    # print(tokenizer.decode(generation_output[0], skip_special_tokens=True))





if __name__ == '__main__':
    main()