import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

count = len(tokenizer.encode("Answer the question based on the context"))

print(count)