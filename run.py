import pandas as pd
from tqdm import tqdm
import collections
import json


# def func_split(row):
#     grams = row.split()
#     res = [" ".join(grams[:3]), grams[-1]]
#     return res

def store_dict(data, path):
    with open(path, 'w') as fw:
        d = json.dumps(data, default=lambda x: list(x) if isinstance(x, set) else x)
        fw.write(d)


if __name__ == "__main__":
    df_sent = pd.read_csv("./data/sentence_cleaned_appended.csv")
    # the list of NUMBER_WORDS_SENTENCE
    len_list = df_sent['NUMBER_WORDS_SENTENCE'].values.tolist()
    # insert length of two unknown words into length list
    len_list.insert(4121, 1)
    len_list.insert(4123, 1)
    print("length of swords", sum(len_list))

    all_words = pd.read_csv("./data/entropy.csv")
    all_words['WORD_pure'] = all_words['WORD_pure'].str.lower()
    # convert all data to str type and get the list of all words
    words = all_words['WORD_pure'].astype(str).values
    print(len(words))

    df = pd.read_csv("./data/ngram-4-norm.tsv", sep='\t')
    vocal = set()
    dic = collections.defaultdict(list)
    for row in df['gram'].values:
        gram = row.split()
        for ch in gram:
            vocal.add(ch)
        dic[" ".join(gram[:3])].append(gram[-1])

    # df['gram'] = df['gram'].apply(lambda row: func_split(row))
    print(len(vocal))
    print(dic["<s> <s> <s>"])
    store_dict(dic, "./data/dic.json")

    # traverse the index
    index = 0
    # initialize a list to store the calculated entropy
    entropy_list = list()
    can_list = []
    # traverse sentence according to the length of sentence
    for sent_len in tqdm(len_list):
        # current sentence
        sent = ['<s>', '<s>', '<s>'] + words[index: index + sent_len].tolist()
        index += sent_len
        before = 1
        for i in range(len(sent) - 3):
            # get the first i words to predict the word i by using first i-1 words
            segment = sent[i:i + 4]
            for idx in range(len(segment)):
                if segment[idx] not in vocal:
                    segment[idx] = '<unk>'
            context_before = " ".join(segment[:3])
            candidate = set(dic[context_before])
            candidate.add(sent[i+3])
            can_list.append(list(candidate))
    print(can_list[:5])
    dic_cand = {}
    for i in range(56410):
        dic_cand[i] = can_list[i]
    store_dict(dic_cand, "./data/candidate.json")
