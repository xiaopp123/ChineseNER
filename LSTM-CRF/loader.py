import re
import os
import math
import codecs
import jieba
import re
import numpy as np
import random
import logging
from conlleval import return_report

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
    """
    对训练数据进行编码
    sentences: 输入训练数据
    char_to_id: 字符与id映射表
    tag_to_id: 标签与id映射表
    lower：是否将大写变成小写
    train：是训练数据还是测试数据
    return: 编码后的数据列表,每条数据包含四部分，第一个文本，第二个是字符id，
            第三是分词后的标记，第四是label
    """
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
    """
    padding数据，创建bath迭代器
    """
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
            targets.append(target + padding)

        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        """
        shuffle: False(default) 是否打乱数据顺序
        训练数据batch的迭代器
        """
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    加载预训练词向量
    emb_path: 文件路径
    id_to_word: 训练数据中的词表
    word_dim: 词向量维度
    old_weights: 原来的词向量，随机初始化的
    return:
    new_weight: [vocab, word_dim], 训练数据中词向量矩阵
    """
    new_weight = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    #加载本地词向量文件
    pre_trained = {}
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
    #为训练数据中的词分配相应的词向量
    n_words = len(id_to_word)
    for i in range(n_words):
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

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def test_ner(results, path):
    """
    ner_results: [batch, ["char true_label pred_label"]
    result_path: save path
    """
    output_file = os.path.join(path, "ner_predict.utf-8")
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)

    return eval_lines

def load_config(path):
    """
    load cinfiguration of the model
    parameters are stored in json format
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def full_to_half(s):
    """
        Convert full-width character to half-width one 
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)

    return ''.join(n)

def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")

    return(s)

def input_from_line(line, char_to_id):
    """
    take sentene data and return an input for training or evaluation function
    """
    line = full_to_half(line) #
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"] for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    
    return inputs

def result_to_json(string, tags):
    """
    tags:format IBES
    """
    item = {"string":string, "entities":[]}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": char, "start": entity_start, "end": idx + 1, "type":tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1

    return item
