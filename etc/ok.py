from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained('./Language_files')


print(tokenizer.vocab_size)
