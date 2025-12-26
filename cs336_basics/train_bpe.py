import regex as re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size: int = 500, text: str = ["hey how are you doing today?"]):
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        self.merges = []

        self.text = text

        self.lookup_table = {}
        self.cache = {}

    def chunk_text(self) -> dict[list[bytes], int]:
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        text_chunks = re.findall(GPT4_SPLIT_PATTERN, self.text)

        freq_table = {}
        for chunk in text_chunks:
            if len(chunk) == 1:
                continue
            chunk_tokens = tuple(chunk.encode("utf-8"))
            freq_table[chunk_tokens] = freq_table.get(chunk_tokens, 0) + 1
        
        return freq_table
    
    
    def iter_bpe(self, freq_table: dict[tuple[int, ...], int], vocab_id: int):

        count_table = {}

        for tokens, freq in freq_table.items():
            if len(tokens) == 2:
                self.cache[tokens] = self.cache.get(tokens, 0) + freq
                continue

            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                count_table[pair] = count_table.get(pair, 0) + freq

        for pair, freq in self.cache.items():
            count_table[pair] = count_table.get(pair, 0) + freq

        pair, _ = max(count_table.items(), key=lambda x: x[1])

        self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))
        
        word_bytes= self.vocab[pair[0]] + self.vocab[pair[1]]
        self.vocab[vocab_id] = word_bytes

        word_str = word_bytes.decode("utf-8", errors="replace")
        self.lookup_table[word_str] = vocab_id

        new_freq_table = {}
        pair0, pair1 = pair

        for tokens, freq in freq_table.items():
            n = len(tokens)
            if n <= 2:
                continue

            i = 0
            out = []
            append = out.append  # 本地绑定，少一次属性查找

            while i < n - 1:
                if tokens[i] == pair0 and tokens[i + 1] == pair1:
                    append(vocab_id)
                    i += 2
                else:
                    append(tokens[i])
                    i += 1

            append(tokens[-1])
            out = tuple(out)

            new_freq_table[out] = new_freq_table.get(out, 0) + freq
        if pair in self.cache:
            del self.cache[pair]

        return new_freq_table


    def train(self) -> None:
        freq_table = self.chunk_text()

        for i in range(256, self.vocab_size):
            freq_table = self.iter_bpe(freq_table,i)
            #print(f"byte length @ {i}: ", len(freq_table))

        return self.vocab, self.merges

if __name__ == "__main__":
    with open("../../text.txt", "r", encoding="utf-8") as f:
        text = f.read()
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{2,}', '\n\n', text)

    vocab_size = 260
    BPE = BPETokenizer(vocab_size, text)
    vocab, merges = BPE.train()
    #print(len(vocab))
    #print(merges)

