import re
import math
import codecs
import jieba
import re

def zero_digits(s):
    return re.sub('\d', '0', s)

def load_sentences(path, lower=True, zeros=False):
    """

    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line or line == '':
            sentences.append(sentence)
            sentence = []
            num += 1
        else:
            if line[0] == " ":
                line = "$" + line[1:]
            word = line.split()
            sentence.append(word)

    if len(sentence) > 0:
        sentences.append(sentence)

    return sentences

def char_mapping(sentences, lower):
    char_dict =  {}
    chars = list()
    for s in sentences:
        chars.extend([w[0].lower() if lower else w[0] for w in s])
    char_dict['<PAD>'] = 1000001
    char_dict['<UNK>'] = 1000000
    for char in chars:
        if char in char_dict:
            char_dict[char] += 1
        else:
            char_dict[char] = 1
    char_to_id = {}
    id_to_char = {}
    for i, key in enumerate(char_dict.keys()):
        char_to_id[key] = i
        id_to_char[i] = key

    return char_dict, char_to_id, id_to_char

def tag_mapping(sentences):
    char_dict =  {}
    chars = list()
    for s in sentences:
        chars.extend([w[-1] for w in s])
        #print(chars)
    for char in chars:
        if char in char_dict:
            char_dict[char] += 1
        else:
            char_dict[char] = 1
    char_to_id = {}
    id_to_char = {}
    for i, key in enumerate(char_dict.keys()):
        char_to_id[key] = i
        id_to_char[i] = key

    return char_dict, char_to_id, id_to_char

def get_seg_feature(sentence):
    seg_feature = []
    for word in jieba.cut(sentence):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)

    return seg_feature

def prepare_data(sentences, char_to_id, tag_to_id, lower=False, train=True):
    none_index = tag_to_id["O"]
    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>'] for w in string]
        seg = get_seg_feature("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, seg, tags])

    return data

class BatchManager(object):
    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)
    
    def sort_and_pad(self, data, batch_size):
        num_batch = (int)(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x : len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size : (i + 1) * batch_size]))

        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        #每个batch的长度不一样
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            target.append(target + padding)

        return [strings, chars, segs, targets]

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)

    n_words = len(id_to_word)
    for i in range(n_word):
        word = id_to_word[i]
        if word in pre_trained:
            new_weight[i] = pre_trained[word]
        elif word.lower() in pre_trained:
            new_weight[i] = pre_trained[word.lower()]
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weight[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]

    return new_weight
