import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
import collections


def func_split(row):
    grams = row.split()
    res = [" ".join(grams[:3]), grams[-1]]
    return res


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

    df = pd.read_csv("./data/ngram-4.tsv", sep='\t', nrows=10000)
    df['gram'] = df['gram'].apply(lambda row: func_split(row))
    print(df.head())

    dic = collections.defaultdict(dict)
    for gram, freq in df.values:
        key1, key2 = gram
        dic[key1][key2] = freq

    model = collections.defaultdict(dict)
    for key1 in dic:
        for key2 in dic[key1]:
            model[key1][key2] = dic[key1][key2] / sum(dic[key1].values())

    # traverse the index
    index = 0
    # initialize a list to store the calculated entropy
    entropy_list = list()
    # traverse sentence according to the length of sentence
    for sent_len in tqdm(len_list):
        # current sentence
        sent = ['<s>', '<s>', '<s>'] + words[index: index + sent_len].tolist()
        index += sent_len
        before = 1
        for i in range(len(sent) - 3):
            # get the first i words to predict the word i by using first i-1 words
            segment = sent[i:i + 4]
            context_before = " ".join(segment[:3])
            print(context_before)
            if context_before in model:
                etp = entropy(list(model[context_before].values()))
            else:
                etp = 0
            entropy_list.append(etp)
    all_words['4gram_entropy'] = entropy_list
    all_words.to_csv("./data/entropy_new.csv", index=False)
