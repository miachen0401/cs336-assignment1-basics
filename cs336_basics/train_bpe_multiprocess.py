from __future__ import annotations

import os
import regex as re
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, List, Iterable
from datetime import datetime

Pair = Tuple[int, int]
FreqTable = Dict[Tuple[int, ...], int]

pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _chunk_list(items: List, n_chunks: int) -> List[List]:
    """Split a list into n_chunks roughly equal parts."""
    if n_chunks <= 1 or not items:
        return [items]

    k, m = divmod(len(items), n_chunks)
    return [
        items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n_chunks)
    ]

def _count_pairs_worker(items: List[Tuple[Tuple[int, ...], int]]) -> Dict[Pair, int]:
    """Count adjacent token-id pairs in a shard of (tokens, freq) items."""
    local: Dict[Pair, int] = {}

    for tokens, freq in items:
        n = len(tokens)
        if n < 2:
            continue

        # local binding for speed
        get = local.get
        for i in range(n - 1):
            p = (tokens[i], tokens[i + 1])
            local[p] = get(p, 0) + freq

    return local

def _merge_counts(dicts: Iterable[Dict[Pair, int]]) -> Dict[Pair, int]:
    """Merge pair-count dicts by summing counts."""
    out: Dict[Pair, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


class BPETokenizer:
    def __init__(self, vocab_size: int = 500,
        input_path: str = "data/txt.txt",
        special_tokens: list[str] | None = None,
        print_interval: int = 10):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] - tokens that should not be split/merged
        '''
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}

        self.merges = []

        self.input_path = input_path

        self.special_tokens = special_tokens or []

        self.print_interval = print_interval
    
    def chunk_text(self,
    ) -> Dict[Tuple[int, ...], int]:
        """Build pre-token frequency table from a text segment."""


        if self.special_tokens:
            delim = "|".join(re.escape(st) for st in self.special_tokens)
            segments = re.split(delim, self.text)
        else:
            segments = [self.text]

        freq: Dict[Tuple[int, ...], int] = {}
        get = freq.get

        for seg in segments:
            for m in re.finditer(pattern, seg):
                chunk = m.group(0)
                if not chunk:
                    continue
                ids = tuple(chunk.encode("utf-8"))
                freq[ids] = get(ids, 0) + 1
        return freq

    
    def iter_bpe(self, freq_table: dict[tuple[int, ...], int], vocab_id: int, num_workers: int):

        count_table = {}

        items = list(freq_table.items())

        if num_workers is None or num_workers <= 1:
            count_table = _count_pairs_worker(items)
        else:
            n_workers = min(num_workers, os.cpu_count() or 1)
            shards = _chunk_list(items, n_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                partials = ex.map(_count_pairs_worker, shards, chunksize=1)
            count_table = _merge_counts(partials)

        #pair = max(count_table.items(), key=lambda kv: (kv[1], kv[0]))[0]
        pair = max(
            count_table.items(),
            key=lambda kv: (kv[1], self.vocab[kv[0][0]], self.vocab[kv[0][1]]),
        )[0]

        self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))

        word_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.vocab[vocab_id] = word_bytes

        #word_str = word_bytes.decode("utf-8", errors="replace")
        #self.lookup_table[word_str] = vocab_id

        new_freq_table = {}
        pair0, pair1 = pair

        for tokens, freq in freq_table.items():
            n = len(tokens)
            if n < 2:
                new_freq_table[tokens] = new_freq_table.get(tokens, 0) + freq
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

            if i == n - 1:
                append(tokens[-1])
            out = tuple(out)

            new_freq_table[out] = new_freq_table.get(out, 0) + freq

        return new_freq_table


    def stream_freq_table(self,
        chunk_size: int = 32 * 1024 * 1024,  # 32MB
        carry_chars: int = 8192,             # 保留尾巴，避免切断 regex token
    ) -> FreqTable:
        """Build freq_table by streaming the file instead of reading it all."""

        freq: FreqTable = {}
        get = freq.get

        file_size = os.path.getsize(self.input_path)
        total_chunks = max(1, math.ceil(file_size / chunk_size))

        if self.special_tokens:
            delim = "|".join(re.escape(st) for st in self.special_tokens)
            split_re = re.compile(delim)
        else:
            split_re = None

        pat_re = re.compile(pattern)

        carry = ""
        chunk_idx = 0

        with open(self.input_path, "r", encoding="utf-8", errors="ignore") as f:
            print(f"Start Processing File {self.input_path}")
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                chunk_idx += 1
                if chunk_idx % self.print_interval == 0:
                    print(f"Processing chunk {chunk_idx}/{total_chunks}")

                text = carry + chunk
                # 留一段尾巴到下一轮（避免正好切在 token 中间）
                if len(text) > carry_chars:
                    carry = text[-carry_chars:]
                    text = text[:-carry_chars]
                else:
                    carry = text
                    continue

                segments = split_re.split(text) if split_re else (text,)
                for seg in segments:
                    for m in pat_re.finditer(seg):
                        tok = m.group(0)
                        if not tok:
                            continue
                        ids = tuple(tok.encode("utf-8"))
                        freq[ids] = get(ids, 0) + 1

            # flush remaining carry
            if carry:
                segments = split_re.split(carry) if split_re else (carry,)
                for seg in segments:
                    for m in pat_re.finditer(seg):
                        tok = m.group(0)
                        if tok:
                            ids = tuple(tok.encode("utf-8"))
                            freq[ids] = get(ids, 0) + 1

        return freq

    def train(self, num_workers: int = 0) -> None:
        freq_table = self.stream_freq_table()

        for i in range(256, self.vocab_size):
            print(f"Training Step {i-255}/{self.vocab_size-256}")
            start_time = datetime.now()
            freq_table = self.iter_bpe(freq_table,i, num_workers)
            end_time = datetime.now()
            print(f"Time taken: {end_time - start_time}")

        return self.vocab, self.merges

if __name__ == "__main__":
    #with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    #with open("../text.txt", "r", encoding="utf-8") as f:
    #    text = f.read()
    #print("file successfully loaded")

    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    BPE = BPETokenizer(vocab_size, input_path, special_tokens, 10)
    vocab, merges = BPE.train(num_workers=4)

