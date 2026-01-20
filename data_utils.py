def tokenize_sentences(sentences, tokenizer):
    return tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True
    )
