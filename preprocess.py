import sys, csv, logging
from transformers import RobertaTokenizer

def replace_bytes(str):
    str = str.replace("Ġ", "")
    str = str.replace("âĢĵ", "–")
    return str

def preprocess(file_path, pretrained_model):
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader) # skip header row
        rows = [row for row in reader]

        prev_sent_id = rows[0][0] # sent_id of first sentence
        sent_words = [] # list of words in each sentence
        sent_strings = [] # raw string of each sentence
        sent_tokenized_idx = [] # list of indices for every token of each word according to RoBERTa tokenization
        sent_first_idx = [] # list of indices for the "first token" of each word according to RoBERTa tokenization
        sent_wlen = [] # list of word lengths in characters
        sent_word_id = [] # list of word_ids, will be converted to proportion later
        sent_len = [] # list of sent_lens, used to convert word_ids to proportions
        sent_prop = []
        sent_nfix, sent_ffd, sent_gpt, sent_trt, sent_fixprop = ([] for _ in range(5)) # five dependent variables

        # sentence-level features
        curr_sent_words, curr_sent_wlen, curr_sent_word_id, curr_sent_nfix, curr_sent_ffd,\
        curr_sent_gpt, curr_sent_trt, curr_sent_fixprop = ([] for _ in range(8))

        for row in rows:
            if "<EOS>" in row[2]: # remove unnecessary <EOS> token
                word = row[2][:-5]
            else:
                word = row[2]

            curr_sent_id = row[0]
            if curr_sent_id == prev_sent_id:
                # populate sentence-level features
                curr_sent_words += [word]
                curr_sent_wlen.append(len(word))
                curr_sent_word_id.append(int(row[1]))
                curr_sent_nfix.append(float(row[3]))
                curr_sent_ffd.append(float(row[4]))
                curr_sent_gpt.append(float(row[5]))
                curr_sent_trt.append(float(row[6]))
                curr_sent_fixprop.append(float(row[7]))
                prev_sent_id = curr_sent_id
            else:
                sent_words.append(curr_sent_words)
                sent_strings.append(" ".join(curr_sent_words))
                sent_wlen.append(curr_sent_wlen)
                sent_word_id.append(curr_sent_word_id)
                sent_len.append(curr_sent_word_id[-1])
                sent_nfix.append(curr_sent_nfix)
                sent_ffd.append(curr_sent_ffd)
                sent_gpt.append(curr_sent_gpt)
                sent_trt.append(curr_sent_trt)
                sent_fixprop.append(curr_sent_fixprop)
                # re-initialize
                curr_sent_words, curr_sent_wlen, curr_sent_word_id, curr_sent_nfix, curr_sent_ffd,\
                curr_sent_gpt, curr_sent_trt, curr_sent_fixprop = ([] for _ in range(8))
                curr_sent_words += [word]
                curr_sent_wlen.append(len(word))
                curr_sent_word_id.append(int(row[1]))
                curr_sent_nfix.append(float(row[3]))
                curr_sent_ffd.append(float(row[4]))
                curr_sent_gpt.append(float(row[5]))
                curr_sent_trt.append(float(row[6]))
                curr_sent_fixprop.append(float(row[7]))
                prev_sent_id = curr_sent_id

        sent_words.append(curr_sent_words)
        sent_strings.append(" ".join(curr_sent_words))
        sent_wlen.append(curr_sent_wlen)
        sent_word_id.append(curr_sent_word_id)
        sent_len.append(curr_sent_word_id[-1])
        sent_nfix.append(curr_sent_nfix)
        sent_ffd.append(curr_sent_ffd)
        sent_gpt.append(curr_sent_gpt)
        sent_trt.append(curr_sent_trt)
        sent_fixprop.append(curr_sent_fixprop)

    for ids, length in zip(sent_word_id, sent_len):
        sent_prop.append([word_id/length for word_id in ids])

    for string, words in zip(sent_strings, sent_words):
        # print(string,words)
        tokenized_ids = tokenizer(string)["input_ids"]
        sent_tokenized_idx.append(tokenized_ids)
        tokenized_list = tokenizer.convert_ids_to_tokens(tokenized_ids)
        # print(tokenized_list)
        curr_idx = 0
        first_idx = []

        for word in words:
            while not word.startswith(replace_bytes(tokenized_list[curr_idx])):
                # print("Evaluating {} and {}".format(tokenized_list[curr_idx].replace('Ġ', ''), word))
                curr_idx += 1
            else:
                # print("Found match: {} and {}".format(tokenized_list[curr_idx].replace('Ġ', ''), word))
                first_idx.append(curr_idx)
                curr_idx += 1
        sent_first_idx.append(first_idx)

    # tokenized idx, valid idx, wlen, prop, five dependent variables
    return [sent_strings, sent_first_idx, sent_wlen, sent_prop, sent_nfix, sent_ffd, sent_gpt, sent_trt, sent_fixprop, sent_words]

def divide(data, dev_size):
    train_size = len(data[0]) - dev_size
    logging.info("Splitting last {} sentences as dev data.".format(dev_size))
    return [part[:train_size] for part in data], [part[train_size:] for part in data]

def oov_indices(train_sent_words, dev_sent_words):
    in_vocab_idx = []
    oov_idx = []
    train_words = set([word for sublist in train_sent_words for word in sublist])
    dev_words = [word for sublist in dev_sent_words for word in sublist]
    curr_idx = 0
    for word in dev_words:
        if word in train_words:
            in_vocab_idx.append(curr_idx)
        else:
            oov_idx.append(curr_idx)
        curr_idx += 1
    return in_vocab_idx, oov_idx

def preprocess_test(file_path, pretrained_model):
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader) # skip header row
        rows = [row for row in reader]

        prev_sent_id = rows[0][0] # sent_id of first sentence
        sent_words = [] # list of words in each sentence
        sent_strings = [] # raw string of each sentence
        sent_tokenized_idx = [] # list of indices for every token of each word according to RoBERTa tokenization
        sent_first_idx = [] # list of indices for the "first token" of each word according to RoBERTa tokenization
        sent_wlen = [] # list of word lengths in characters
        sent_word_id = [] # list of word_ids, will be converted to proportion later
        sent_len = [] # list of sent_lens, used to convert word_ids to proportions
        sent_prop = []

        # sentence-level features
        curr_sent_words, curr_sent_wlen, curr_sent_word_id = ([] for _ in range(3))

        for row in rows:
            if "<EOS>" in row[2]: # remove unnecessary <EOS> token
                word = row[2][:-5]
            else:
                word = row[2]

            curr_sent_id = row[0]
            if curr_sent_id == prev_sent_id:
                # populate sentence-level features
                curr_sent_words += [word]
                curr_sent_wlen.append(len(word))
                curr_sent_word_id.append(int(row[1]))
                prev_sent_id = curr_sent_id
            else:
                sent_words.append(curr_sent_words)
                sent_strings.append(" ".join(curr_sent_words))
                sent_wlen.append(curr_sent_wlen)
                sent_word_id.append(curr_sent_word_id)
                sent_len.append(curr_sent_word_id[-1])
                # re-initialize
                curr_sent_words, curr_sent_wlen, curr_sent_word_id = ([] for _ in range(3))
                curr_sent_words += [word]
                curr_sent_wlen.append(len(word))
                curr_sent_word_id.append(int(row[1]))
                prev_sent_id = curr_sent_id

        sent_words.append(curr_sent_words)
        sent_strings.append(" ".join(curr_sent_words))
        sent_wlen.append(curr_sent_wlen)
        sent_word_id.append(curr_sent_word_id)
        sent_len.append(curr_sent_word_id[-1])

    for ids, length in zip(sent_word_id, sent_len):
        sent_prop.append([word_id/length for word_id in ids])

    for string, words in zip(sent_strings, sent_words):
        # print(string,words)
        tokenized_ids = tokenizer(string)["input_ids"]
        sent_tokenized_idx.append(tokenized_ids)
        tokenized_list = tokenizer.convert_ids_to_tokens(tokenized_ids)
        # print(tokenized_list)
        curr_idx = 0
        first_idx = []

        for word in words:
            while not word.startswith(replace_bytes(tokenized_list[curr_idx])):
                # print("Evaluating {} and {}".format(tokenized_list[curr_idx].replace('Ġ', ''), word))
                curr_idx += 1
            else:
                # print("Found match: {} and {}".format(tokenized_list[curr_idx].replace('Ġ', ''), word))
                first_idx.append(curr_idx)
                curr_idx += 1
        sent_first_idx.append(first_idx)

    # tokenized idx, valid idx, wlen, prop, five dependent variables
    return [sent_strings, sent_first_idx, sent_wlen, sent_prop, sent_words]