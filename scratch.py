from generate import simple_generate_with_tokenization, MyLlamaTokenizer
from transformers import AutoTokenizer, AutoModel
from model.model_utils import load_hf_model


def main():
    model = load_hf_model('storage/models/hugging_face_models/Meta-Llama-3-8B', load_in_8bit=True)
    tokenizer = MyLlamaTokenizer('storage/models/hugging_face_models/Meta-Llama-3-8B')

    out = simple_generate_with_tokenization(
        model,
        input_str='hello 1 2 3 4',
        tokenizer=tokenizer,
        max_new_tokens=10,
    )

    print(out)

if __name__ == '__main__':
    main()