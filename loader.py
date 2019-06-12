import re
import codecs

def zero_digits(s):
    return re.sub('\d', '0', s)

def load_sentences(path, lower=True, zeros=False):
    """

    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf-8'):
        num += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            sentences.append(sentence)

            sentnce = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
            word = line.split()
            sentences.append(word)
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
        chars.extend([w[-1].lower() if lower else w[-1] for w in s])
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
