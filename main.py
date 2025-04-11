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


import re

word_re = re.compile("^(\\d+)\t([^\\t]+)\\t([^\\t]+)\\t([A-Z]+)\\t(.*)$")

line = "7	Якобсона	Якобсон	PROPN	_	Animacy=Anim|Case=Gen|Gender=Masc|Number=Sing	6	flat:name	6:flat:name	SpaceAfter=No"
m = word_re.match(line)



with open("/x/cent") as f:
    lines = f.readlines()
sents = []
sent_words = []
sent = None
words = []
for l in lines:
    if l.startswith("# text ="):
        if sent is not None:
            sents.append(sent)
            sent_words.append(words)
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
cent = None
words = None
assert len(sents) == len(sent_words)

print(len(sents))
