# Licensed under the Apache License, Version 2.0 (the "License");
# Ported from LogGPT utils/vocab.py (https://github.com/nokia/LogGPT, commit 28b15f9), which itself
# derives from torchtext's Vocab. Only the parts LogGPT uses are kept; the token order is unchanged:
# 5 special tokens, then the training tokens sorted by frequency (descending), ties alphabetically.
from collections import Counter

SPECIALS = ["<pad>", "<unk>", "<eos>", "<bos>", "<mask>"]


class WordVocab:
    pad_index, unk_index, eos_index, bos_index, mask_index = 0, 1, 2, 3, 4

    def __init__(self, texts):
        """texts: iterable of token lists (here: template ids as strings)."""
        counter = Counter()
        for line in texts:
            for word in line:
                counter[word] += 1
        self.freqs = counter
        self.itos = list(SPECIALS)
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        for word, _ in words_and_frequencies:
            self.itos.append(word)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @property
    def n_keys(self):
        return len(self.itos) - len(SPECIALS)

    def forward(self, sentence):
        """sliding_window=False branch of the original: <bos> + ids + <eos>; unseen tokens -> <unk>."""
        return [self.bos_index] + [self.stoi.get(word, self.unk_index) for word in sentence] + [self.eos_index]
