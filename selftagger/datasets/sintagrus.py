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
