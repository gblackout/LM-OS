from generate import simple_generate_with_tokenization, MyLlamaTokenizer, simple_generate
from transformers import AutoTokenizer, AutoModel
from model.model_utils import load_hf_model
from model.compressor_retriever import CompressorRetriever, CompressorRetrieverConfig
import torch
from copy import deepcopy
from utils.misc import jload, jdump, PCYAN_L, PCYAN_R, PGREEN_L, PGREEN_R
from tqdm import tqdm
from peft import PeftModel


def main(ispeft=False, few_shot=True):
    model = load_hf_model(
        'storage/models/hugging_face_models/Meta-Llama-3.1-8B-Instruct',
        load_in_8bit=True,
        use_flash_attn=True
    )
    tokenizer = MyLlamaTokenizer('storage/models/hugging_face_models/Meta-Llama-3.1-8B-Instruct')
    tokenizer.llama_tokenizer.padding_side = "left"  # Allow batched inference


    if not ispeft:
        prompt = """Given a set of premises and a hypothesis, answer if the hypothesis
agrees with the premises [Agree],
contradicts with the premises [Contradict],
or neutral with respect to the premises [Uncertain]."""

        ret_data = jload('data/ret_combined_data_550.json')
        ret_test_data = [e for e in ret_data if e['src'] == 'test']
        cnt = 0

        for sample in tqdm(ret_test_data):
            if few_shot:
                main_p = '\n\n===\n\n'.join(sample['ctx_ls']) + '\n\n===\n\n' + sample['input']
            else:
                main_p = sample['input']

            p = f"{prompt}\n\n{main_p}"

            inputs = tokenizer(p, my_token2llama_token=True)
            input_ids = inputs['input_ids'].to('cuda')
            input_id_len = len(input_ids[0])

            full_ids = simple_generate(
                model,
                input_ids,
                max_new_tokens=300,
                generation_config=None
            )
            resp_ids = full_ids[input_id_len:]
            pred_str = tokenizer.decode(resp_ids, skip_special_tokens=True)

            if 'Agree' in sample['label']:
                gt_ans = 'Agree'
            elif 'Contradict' in sample['label']:
                gt_ans = 'Contradict'
            else:
                gt_ans = 'Uncertain'

            if 'Agree' in pred_str:
                pred = 'Agree'
            elif 'Contradict' in pred_str:
                pred = 'Contradict'
            elif 'Uncertain' in pred_str:
                pred = 'Uncertain'
            else:
                pred = ''

            hit = gt_ans == pred
            cnt += 1 if hit else 0

            print(
                f"{PCYAN_L}{sample['label']}{PCYAN_R}\n\n"
                f"{PGREEN_L}{pred_str}{PGREEN_R}\n\n===\n\n"
                f"hit {hit}\n\n===\n"
            )

            sample['ret-base-pred-few-shot'] = pred_str

        print(cnt / len(ret_test_data))
        jdump(ret_test_data, 'data/ret_demo_base_pred_few_shot.json')

    if ispeft:

        model = PeftModel.from_pretrained(
            model,
            'logs/ret_combined_data-2024-08-25-21-55-16/checkpoint-30',
            torch_dtype=torch.float16
        )
        model.to('cuda')

        cr_model = CompressorRetriever.from_pretrained(
            'logs/ret_combined_data-2024-08-25-21-55-16/checkpoint-30',
            base_causallm=model
        )
        cr_model.to('cuda')

        def ret_demo_tokenize(sample):
            main_dict = tokenizer(
                [
                    [sample['input'], 'False'],
                    [sample['label'], 'True'],
                ],
                add_bos=False,
                add_eos=False,
                single_dim=True,
                my_token2llama_token=True
            )
            ctx_ls = [s.replace(tokenizer.MY_EOS_TOKEN, tokenizer.llama_tokenizer.eos_token) for s in sample['ctx_ls']]
            ctx_ids = tokenizer.llama_tokenizer(
                ctx_ls,
                padding=True,
                pad_to_multiple_of=8,
                add_special_tokens=False,
                return_tensors='pt'
            )
            main_dict['ctx_ids'] = ctx_ids['input_ids']
            main_dict['ctx_attention_masks'] = ctx_ids['attention_mask']
            main_dict['relevant_ctx_ids'] = torch.tensor(sample['relevant_ctx_ids'])
            del main_dict['labels']

            for k,v in main_dict.items():
                if isinstance(v, torch.Tensor):
                    main_dict[k] = v[None, ...].to('cuda')

            return main_dict

        ret_data = jload('data/ret_combined_data_550.json')
        ret_test_data = [e for e in ret_data if e['src'] == 'train']

        for sample in tqdm(ret_test_data):
            main_dict = ret_demo_tokenize(sample)
            cr_model.eval()

            with torch.no_grad():
                generation_output, topk = cr_model.demo_retrieval_generation(
                    generation_config=None,
                    max_new_tokens=100,
                    **main_dict
                )

            # print(generation_output[0])
            print(topk[0])
            print(
                f"{PCYAN_L}{sample['label']}{PCYAN_R}\n\n"
                f"{PGREEN_L}{tokenizer.decode(generation_output[0], skip_special_tokens=True)}{PGREEN_R}"
            )
            print('===')


    # compress_test = jload('data/compress_test_folio_300.json')
    #
    # for sample in tqdm(compress_test):
    #     s = f'{sample["ctx"]}\n\n{sample["input"]}'
    #     print(f'{PCYAN_L}{sample["label"]}{PCYAN_R}')
    #     inputs = tokenizer(s, my_token2llama_token=True)
    #     input_ids = inputs['input_ids'].to('cuda')
    #     input_id_len = len(input_ids[0])
    #
    #     full_ids = simple_generate(
    #         model,
    #         input_ids,
    #         max_new_tokens=10,
    #         generation_config=None
    #     )
    #     resp_ids = full_ids[input_id_len:]
    #     print(f'{PGREEN_L}{tokenizer.decode(resp_ids, skip_special_tokens=True)}{PGREEN_R}')
    #
    #     sample['2shot-pred'] = tokenizer.decode(resp_ids, skip_special_tokens=True)
    # jdump(compress_test, 'data/compress_test_folio_300_2shot-pred.json')

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