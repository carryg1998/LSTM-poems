import numpy as np

start_mark = 'B'
end_mark = 'E'

def load_txt(file_name):
    res = {}
    txt = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            try:
                content = start_mark + line + end_mark
                txt.append(content)
            except:
                pass

    all_vocab = [word for line in txt for word in line]
    vocabs = list(set(all_vocab))
    vocabs.append(" ")
    vocabs = list(set(vocabs))
    vocabs = sorted(vocabs)
    
    count = len(vocabs)
    int_to_word = dict(zip(range(count), vocabs))
    word_to_int = dict(zip(vocabs, range(count)))
    
    encoded = [list(map(lambda word: word_to_int.get(word, count), char)) for char in txt]
    
    res["encoded"] = encoded
    res["all_vocab"] = all_vocab
    res["vocabs"] = vocabs
    res["count"] = count
    res["int_to_word"] = int_to_word
    res["word_to_int"] = word_to_int
    return res

def load_lrc(file_name):
    res = {}
    txt = []
    with open(file_name, "r", encoding="utf-8") as f:
        lrc = ""
        for line in f.readlines():
            try:
                if 'E' in line:
                    lrc = lrc + line
                    txt.append(lrc)
                    lrc = ""
                else:
                    lrc = lrc + line
            except:
                pass

    all_vocab = [word for line in txt for word in line]
    vocabs = list(set(all_vocab))
    vocabs.append(" ")
    vocabs = list(set(vocabs))
    vocabs = sorted(vocabs)
    
    count = len(vocabs)
    int_to_word = dict(zip(range(count), vocabs))
    word_to_int = dict(zip(vocabs, range(count)))
    
    encoded = [list(map(lambda word: word_to_int.get(word, count), char)) for char in txt]
    
    res["encoded"] = encoded
    res["all_vocab"] = all_vocab
    res["vocabs"] = vocabs
    res["count"] = count
    res["int_to_word"] = int_to_word
    res["word_to_int"] = word_to_int
    return res


def get_batches(batch_size, word_vec):
    x_batches = []
    y_batches = []
    n = len(word_vec)//batch_size
    for i in range(n):
        b_index = i * batch_size
        e_index = b_index + batch_size
        
        batch = word_vec[b_index:e_index]
        max_len = max(map(len,word_vec))
        x_data = np.full((batch_size, max_len), 1, np.int32)
        for j in range(batch_size):
            for k in range(len(batch[j])):
                x_data[j][k] = batch[j][k]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches