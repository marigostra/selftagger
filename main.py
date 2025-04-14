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
from torch.nn.functional import normalize as norm
import re

data = "sent"
learning_rate = 0.2
epochs = 50
word_re = re.compile("^(\\d+)\t([^\\t]+)\\t([^\\t]+)\\t([A-Z]+)\\t(.*)$")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device ", device)

def compare(pred, target):
    if pred.shape != target.shape:
        raise BaseException("Shape mismatch")
    match_count = 0
    for p, t in zip(pred, target):
        p_max = p.topk(1).indices[0]
        t_max = t.topk(1).indices[0]
        if p_max == t_max:
            match_count = match_count + 1
    return float(match_count) / len(target)

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
            words.append({"word": self.dict_map[word], "pos": self.pos_dict_map[pos]})
            assert len(words) == int(r.group(1)), str(len(words)) + " not equals " + r.group(1)
        assert len(self.sents) == len(self.words)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent = self.words[index]
        input = torch.zeros(len(sent), dtype=torch.long)
        for token_index, token in enumerate(sent):
            input[token_index] = token["word"]

        output = torch.zeros((len(sent), len(self.pos_dict)))
        for token_index, token in enumerate(sent):
            output[token_index][token["pos"]] = 1

        return (input, output)


class Model(nn.Module):
    def __init__(self, dict_size, embed_dim, atten_dim, num_pos, num_heads):
        super().__init__()
        self.embed = nn.Embedding(dict_size, embed_dim)
        self.ln_query = nn.Linear(embed_dim, atten_dim)
        self.ln_key = nn.Linear(embed_dim, atten_dim)
        self.ln_value = nn.Linear(embed_dim, atten_dim)
        self.atten = MHAtten(atten_dim, num_heads)
        self.ln_pos = nn.Linear(atten_dim, num_pos)

    def forward(self, x):
        e = self.embed(x)
        q = self.ln_query(e)
        k = self.ln_key(e)
        v = self.ln_value(e)
        atten, atten_weight = self.atten(q, k, v)        
        return norm(self.ln_pos(atten), p=1, dim=1)

def train(m, train_set, loss_fn, opt):
    step = 0
    matching_sum = 0.0
    for input, target in train_set:
        pred = m(input.to(device))
        loss = loss_fn(pred, target.to(device))
        matching = compare(pred.to("cpu"), target) * 100
        matching_sum = matching_sum + matching
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 3)
        opt.step()
        opt.zero_grad()
        step = step + 1
    return matching_sum / step 

d = Dataset(data)
train_set, test_set = torch.utils.data.random_split(d, [.85, .15], generator=torch.Generator(device="cpu").manual_seed(2024))
m = Model(len(d.dict), 1024, 1024, len(d.pos_dict), 8).to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(m. parameters(), lr=learning_rate)

for i in range(epochs):
    print("Epoch ", i)
    print(train(m, train_set, loss_fn, opt))
