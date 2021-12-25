#!/usr/bin/env python
# coding: utf-8
import random
from transformers import BertTokenizerFast
# tokenizer = BertTokenizerFast('/home/share/Models/bert-large-cased/vocab.txt')
# tokenizer.encode

class ConLL_dataloader(object):
    def __init__(self,data_path, bert_vocab='/home/share/Models/bert-large-cased/vocab.txt'):
        self.data_path = data_path
        self.sentences, self.tags = self.load_data(self.data_path)
        self.tokenizer = BertTokenizerFast(bert_vocab)

    def load_data(self, data_path):
        sentences = []
        tags = []
        with open(data_path, 'r') as f:
            sentence = []
            tag = []
            for line in f:
                if line != '\n':
                    line = line.strip().split(' ')
                    sentence.append(line[0])
                    tag.append(line[3])
                else:
                    if len(sentence) == 0:
                        continue
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []
        return sentences, tags

    def encode_bert(self, sentences, tags):
        tokenizer = self.tokenizer
        sentences_token_idx = []
        tags_bert = []
        for i in range(len(sentences)):
            one_sent = []
            one_sent_tag = []
            for j in range(len(sentences[i])):
                one_token_idx = tokenizer.encode(sentences[i][j], add_special_tokens=False)
                one_sent += one_token_idx
                if len(one_token_idx) == 1:
                    one_sent_tag.append(tags[i][j])
                else:
                    one_tag = tags[i][j]
                    if one_tag.startswith('I-'):
                        one_sent_tag += [one_tag]*len(one_token_idx)
                    elif one_tag.startswith('B-'):
                        one_sent_tag.append(one_tag)
                        one_sent_tag += ['I-'+one_tag.split('-')[1]]*(len(one_token_idx)-1)
                    elif one_tag.startswith('O'):
                        one_sent_tag += ['O']*len(one_token_idx)
                    else:
                        print(one_tag)
                        raise ValueError
            assert len(one_sent) == len(one_sent_tag)
            sentences_token_idx.append(one_sent)
            tags_bert.append(one_sent_tag)
        return sentences_token_idx, tags_bert

    def __len__(self):
        return len(self.sentences)

    def pad(self, sentences, tags, mode='word'):
        """
        args:
            mode: idx for BERT token index.
        """
        max_len = -100
        for i in sentences:
            if max_len < len(i):
                max_len = len(i)
        mask = [[1]*len(one)+[0]*(max_len-len(one)) for one in sentences]
        if mode == 'word':
            sentences = [one+['[PAD]']*(max_len-len(one)) for one in sentences]
        elif mode == 'idx':
            sentences = [one + [0] * (max_len - len(one)) for one in sentences]
        tags = [one+['B-PAD']*(max_len-len(one)) for one in tags]
        return sentences, tags, mask

    def add_special_token(self, sentences_token_idx, tags):
        # For BERT
        sentences_token_idx = [[101]+one+[102] for one in sentences_token_idx]
        tags = [['B-[CLS]']+one+['B-[SEP]'] for one in tags]
        return sentences_token_idx, tags

    def shuffle(self):
        zipped = list(zip(self.sentences, self.tags))
        random.shuffle(zipped)
        sentences_sf = []
        tags_sf = []
        for i in range(len(zipped)):
            sentences_sf.append(zipped[i][0])
            tags_sf.append(zipped[i][1])
        self.sentences = sentences_sf
        self.tags = tags_sf

class AGAC_dataloader(object):
    def __init__(self, data_path, bert_vocab='/home/share/Models/bert-large-cased/vocab.txt'):
        self.data_path = data_path
        self.sentences, self.tags = self.load_data(self.data_path)
        self.tokenizer = BertTokenizerFast(bert_vocab)

    def load_data(self, data_path):
        sentences = []
        tags = []
        with open(data_path, 'r') as f:
            sentence = []
            tag = []
            for line in f:
                if line != '\n':
                    line = line.strip().split('\t')
                    sentence.append(line[0])
                    tag.append(line[1])
                else:
                    if len(sentence) == 0:
                        continue
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []

        return sentences, tags

    def encode_bert(self, sentences, tags):
        tokenizer = self.tokenizer
        sentences_token_idx = []
        tags_bert = []
        for i in range(len(sentences)):
            one_sent = []
            one_sent_tag = []
            for j in range(len(sentences[i])):
                one_token_idx = tokenizer.encode(sentences[i][j], add_special_tokens=False)
                one_sent += one_token_idx
                if len(one_token_idx) == 1:
                    one_sent_tag.append(tags[i][j])
                else:
                    one_tag = tags[i][j]
                    if one_tag.startswith('I-'):
                        one_sent_tag += [one_tag] * len(one_token_idx)
                    elif one_tag.startswith('B-'):
                        one_sent_tag.append(one_tag)
                        one_sent_tag += ['I-' + one_tag.split('-')[1]] * (len(one_token_idx) - 1)
                    elif one_tag.startswith('O'):
                        one_sent_tag += ['O'] * len(one_token_idx)
                    else:
                        print(one_tag)
                        raise ValueError
            assert len(one_sent) == len(one_sent_tag)
            sentences_token_idx.append(one_sent)
            tags_bert.append(one_sent_tag)
        return sentences_token_idx, tags_bert

    def __len__(self):
        return len(self.sentences)

    def pad(self, sentences, tags, mode='word'):
        """
        args:
            mode: idx for BERT token index.
        """
        max_len = -100
        for i in sentences:
            if max_len < len(i):
                max_len = len(i)
        mask = [[1] * len(one) + [0] * (max_len - len(one)) for one in sentences]
        if mode == 'word':
            sentences = [one + ['[PAD]'] * (max_len - len(one)) for one in sentences]
        elif mode == 'idx':
            sentences = [one + [0] * (max_len - len(one)) for one in sentences]
        tags = [one + ['B-PAD'] * (max_len - len(one)) for one in tags]
        return sentences, tags, mask

    def add_special_token(self, sentences_token_idx, tags):
        # For BERT
        sentences_token_idx = [[101] + one + [102] for one in sentences_token_idx]
        tags = [['B-[CLS]'] + one + ['B-[SEP]'] for one in tags]
        return sentences_token_idx, tags

    def shuffle(self):
        zipped = list(zip(self.sentences, self.tags))
        random.shuffle(zipped)
        sentences_sf = []
        tags_sf = []
        for i in range(len(zipped)):
            sentences_sf.append(zipped[i][0])
            tags_sf.append(zipped[i][1])
        self.sentences = sentences_sf
        self.tags = tags_sf






    # def __iter__(self):
    #
    #     pass

if __name__ == "__main__":
    data_path = '../data/ConLL2003/train.txt'
    dataloader = ConLL_dataloader(data_path)
    dataloader.shuffle()
    batch_sent,batch_tag = dataloader.sentences[:8],dataloader.tags[:8]
    # batch_sent, batch_tag, batch_mask = dataloader.pad(batch_sent,batch_tag)
    print(batch_sent)
    # print(batch_tag)
    # print(batch_mask)
    # dataloader.encode_bert()
    # batch_sent, batch_tag = dataloader.sentences_token_idx[:8], dataloader.tags_bert[:8]
    # print(batch_sent)
    # print(batch_tag)