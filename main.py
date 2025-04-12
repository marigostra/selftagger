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
            words.append(r.group(2))
            assert len(words) == int(r.group(1)), str(len(words)) + " not equals " + r.group(1)
        assert len(self.sents) == len(self.words)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.ln_query = nn.Linear(self.input_dim, self.output_dim)
        self.ln_key = nn.Linear(self.input_dim, self.output_dim)
        self.ln_value = nn.Linear(self.input_dim, self.output_dim)
        self.atten = MHAtten(self.output_dim, self.num_heads)

    def forward(self, x):
        q = self.ln_query(x)
        k = self.ln_key(x)
        v = self.ln_value(x)
        return self.atten(q, k, v)        

m = Model(128, 64, 8)
