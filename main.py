# Copyright 2025 Michael Pozhidaev <msp@luwrain.org>
#
# This file is part of SelfTagger.
#
# SelfTagger is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# SelfTagger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

import torch
from torch import tensor
import torch.nn as nn
from torch.nn import MultiheadAttention as MHAtten
import re

word_re = re.compile("^(\\d+)\t([^\\t]+)\\t([^\\t]+)\\t([A-Z]+)\\t(.*)$")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        self.dict = []
        self.dict_map = dict()
        self.pos_dict = []
        self.pos_dict_map = dict()
        self.sents = []
        self.words = []
        with open(file_name) as f:
            lines = f.readlines()
        sent = None
        words = []
        for l in lines:
            if l.startswith("# text ="):
                if sent is not None:
                    self.sents.append(sent)
                    self.words.append(words)
                sent = l[len("# text = "):]
                words = []
                continue
            if l.startswith("#"):
                continue
            r = word_re.match(l)
            if not r:
                continue
            word = r.group(2)
            lemma = r.group(3)
            pos = r.group(4)
            if not word in self.dict_map.keys():
                self.dict.append(word)
                self.dict_map[word] = len(self.dict) - 1
            if not pos in self.pos_dict_map.keys():
                self.pos_dict.append(pos)
                self.pos_dict_map[pos] = len(self.pos_dict) - 1
            words.append({word: self.dict_map[word]})
            assert len(words) == int(r.group(1)), str(len(words)) + " not equals " + r.group(1)
        assert len(self.sents) == len(self.words)

class Model(nn.Module):
    def __init__(self, dict_size, embed_dim, atten_dim, num_heads):
        super().__init__()
        self.embed = nn.Embedding(dict_size, embed_dim)
        self.ln_query = nn.Linear(embed_dim, atten_dim)
        self.ln_key = nn.Linear(embed_dim, atten_dim)
        self.ln_value = nn.Linear(embed_dim, atten_dim)
        self.atten = MHAtten(atten_dim, num_heads)

    def forward(self, x):
        e = self.embed(x)
        q = self.ln_query(e)
        k = self.ln_key(e)
        v = self.ln_value(e)
        return self.atten(q, k, v)        

d = Dataset("/x/sent")
for i in d.pos_dict_map.values():
    print(i)
#m = Model(128, 64, 8)
