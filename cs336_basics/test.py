if __name__ == "__main__":
    with open("../data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
        text = f.read()
    
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    
    for k, v in vocab.items():
        print(k, v)