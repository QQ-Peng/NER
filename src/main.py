import json, copy, os
import numpy as np
import random
from model import LSTM_Model, LSTM_CRF_Model, BERT_LSTM_CRF_Model,BERT_CRF_Model
from dataloader import ConLL_dataloader, AGAC_dataloader
import torch.tensor as tensor
from torch import nn, optim
import torch
from seqeval.metrics import classification_report
from conlleval import evaluate
from seqeval.metrics import f1_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

def build_model(model_used:str='bert_lstm_crf',word2idx=None,tag2idx=None,bert_path=None):
    if model_used == 'lstm':
        ner_model = LSTM_Model(len(word2idx), 1024, 1024, len(tag2idx), 2)
    elif model_used == 'lstm_crf':
        ner_model = LSTM_CRF_Model(len(word2idx), 1024, 1024, len(tag2idx), 2)
    elif model_used in ['bert_lstm_crf','bert_crf']:
        tag2idx['B-[CLS]'] = len(tag2idx)
        tag2idx['B-[SEP]'] = len(tag2idx)
        assert len(set(tag2idx.keys())) == len(set(tag2idx.values()))
        if model_used == 'bert_lstm_crf':
            ner_model = BERT_LSTM_CRF_Model(1024, len(tag2idx), 2, 'mean', True, bert_path)
        elif model_used == 'bert_crf':
            ner_model = BERT_CRF_Model(len(tag2idx), bert_path)
    else:
        raise ValueError
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return ner_model,tag2idx,idx2tag

def build_data(sentences, tags ,model_used, dataloader, tag2idx,word2idx):
    if model_used in ['lstm', 'lstm_crf']:
        sent, tags, mask = dataloader.pad(sentences, tags)
        sent_idx = []
        tag_idx = []
        for i in range(len(sent)):
            sent_idx.append([word2idx[word] for word in sent[i]])
            tag_idx.append([tag2idx[tag] for tag in tags[i]])
    elif model_used in ['bert_lstm_crf','bert_crf']:
        sent_idx, tags = dataloader.encode_bert(sentences, tags)
        sent_idx, tags = dataloader.add_special_token(sent_idx, tags)
        sent_idx, tags, mask = dataloader.pad(sent_idx, tags, 'idx')
        tag_idx = []
        for i in range(len(tags)):
            tag_idx.append([tag2idx[t] for t in tags[i]])
    return sent_idx, tag_idx, mask

def remove_special_token_in_tags_for_true(tag_idx, mask,idx2tag,tag2idx,model_used='bert_lstm_crf'):
    # remove "PAD", "[CLS]", "[SEP]"
    if model_used in ['bert_lstm_crf','bert_crf']:
        Tag_filter = []
        for m in range(len(tag_idx)):
            one_sent_tag = []
            for n in range(len(tag_idx[m])):
                if mask[m][n] != 0 and tag_idx[m][n] not in [tag2idx['B-[CLS]'], tag2idx['B-[SEP]']]:
                    one_sent_tag.append(idx2tag[tag_idx[m][n]])
            if len(one_sent_tag) == 0:
                print(mask[m])
            assert len(one_sent_tag) != 0
            Tag_filter.append(one_sent_tag)
    elif model_used in ['lstm','lstm_crf']:
        Tag_filter = []
        for m in range(len(tag_idx)):
            one_sent_tag = []
            for n in range(len(tag_idx[m])):
                if mask[m][n] != 0:
                    one_sent_tag.append(idx2tag[tag_idx[m][n]])
            assert len(one_sent_tag) != 0
            Tag_filter.append(one_sent_tag)
    return Tag_filter



def remove_special_token_in_tags_for_predicted(pred_tag_idx,mask,idx2tag,model_used='bert_lstm_crf'):
    # remove "PAD", "[CLS]", "[SEP]"

    Tag_filter = []
    for i in range(len(mask)):
        if 0 in mask[i]:
            one_sent_tag_idx = pred_tag_idx[i][:mask[i].index(0)]
        else:
            one_sent_tag_idx = pred_tag_idx[i]
        if model_used in ['bert_lstm_crf','bert_crf']:
            one_sent_tag_idx.pop(0)
            one_sent_tag_idx.pop(-1)
        one_sent_tag = [idx2tag[idx] for idx in one_sent_tag_idx]
        Tag_filter.append(one_sent_tag)


    return Tag_filter

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    data_used = 'AGAC'
    model_used = 'bert_crf'
    bert_path = '/home/share/Models/biobert-base-cased/'
    if model_used in ['lstm', 'lstm_crf']:
        learning_rate = 1e-03
    elif 'bert' in model_used:
        if 'base' in bert_path:
            learning_rate = 1e-04
        # elif 'bio' in bert_path:
        #     learning_rate = 1e-03
        elif 'large' in bert_path:
            learning_rate = 1e-05

    load_from_checkpoint = False

    word2idx = json.load(open(f'../map/{data_used}/word2idx.json', 'r'))
    tag2idx = json.load(open(f'../map/{data_used}/tag2idx.json', 'r'))
    ner_model, tag2idx, idx2tag = build_model(model_used, word2idx,tag2idx,bert_path)
    max_f1score = -100
    if load_from_checkpoint:
        print("load from checkpoint.")
        if os.path.exists(f"../model/{model_used}_{data_used}_f1.txt"):
            max_f1score = float(open(f"../model/{model_used}_{data_used}_f1.txt",'r').read().strip())
        ner_model.load_state_dict(torch.load(f"../model/{model_used}_{data_used}.pkl"))
    # print(ner_model)
    # exit()
    train_data_path = f'../data/{data_used}/train.txt'
    dev_data_path = f'../data/{data_used}/dev.txt'
    train_dataloader = ConLL_dataloader(train_data_path) if data_used=='ConLL2003' else AGAC_dataloader(train_data_path,bert_path+'/vocab.txt')
    train_dataloader.shuffle()
    dev_dataloader = ConLL_dataloader(dev_data_path) if data_used=='ConLL2003' else AGAC_dataloader(dev_data_path,bert_path+'/vocab.txt')
    dev_sent = dev_dataloader.sentences
    dev_tag = dev_dataloader.tags
    if model_used in ['lstm','lstm_crf']:
        trueTag = copy.deepcopy(dev_tag)
    dev_sentence_idx, dev_tag_idx, dev_mask = build_data(dev_sent,dev_tag,model_used,dev_dataloader,tag2idx,word2idx)
    dev_sentence_idx = tensor(dev_sentence_idx, requires_grad=False).to('cuda')
    dev_tag_idx = tensor(dev_tag_idx, requires_grad=False).to('cuda')
    dev_mask = tensor(dev_mask,requires_grad=False).to('cuda')
    assert dev_sentence_idx.shape==dev_tag_idx.shape==dev_mask.shape
    if model_used in ['bert_lstm_crf','bert_crf']:
        trueTag = remove_special_token_in_tags_for_true(dev_tag_idx.tolist(),dev_mask.tolist(),idx2tag,tag2idx,model_used)
    # print(dev_sentence_idx.shape)
    ner_model.to('cuda')

    lossFunction = nn.NLLLoss(reduction='mean')
    optimizer = optim.Adam(ner_model.parameters(), lr=learning_rate, betas=(0.9,0.999),eps=1e-12,weight_decay=1e-5,amsgrad=False)
    # optimizer = optim.SGD(ner_model.parameters(), lr=1e-4)
    step = 16


    for j in range(25):
        train_dataloader.shuffle()
        n = 0
        n_batch = 0
        adjust_learning_rate(optimizer, j, learning_rate)
        while n < len(train_dataloader):
            sentence = train_dataloader.sentences[n:n+step]
            tag = train_dataloader.tags[n:n+step]

            if model_used in ['lstm', 'lstm_crf']:
                sentence, tag, mask = train_dataloader.pad(sentence, tag)
                sentence_idx = []
                tag_idx = []
                for i in range(len(sentence)):
                    sentence_idx.append([word2idx[word] for word in sentence[i]])
                    tag_idx.append([tag2idx[t] for t in tag[i]])
            elif model_used in ['bert_lstm_crf','bert_crf']:
                sentence_idx, tag = train_dataloader.encode_bert(sentence, tag)
                sentence_idx, tag = train_dataloader.add_special_token(sentence_idx, tag)
                sentence_idx, tag, mask = train_dataloader.pad(sentence_idx, tag, 'idx')
                tag_idx = []
                for i in range(len(tag)):
                    tag_idx.append([tag2idx[t] for t in tag[i]])



            sentence_idx = tensor(sentence_idx).to('cuda')
            tag_idx = tensor(tag_idx).to('cuda')
            mask = tensor(mask).to('cuda')

            if model_used == 'lstm':
                tagScore = ner_model(sentence_idx).view(-1, len(tag2idx))
                loss = lossFunction(tagScore, tag_idx.view(-1))
            elif model_used in ['lstm_crf','bert_lstm_crf','bert_crf']:
                loss, emission = ner_model(sentence_idx, tag_idx, mask)
                loss = -loss
            # elif model_used == 'bert_lstm_crf':
            #     loss, emission = ner_model(sentence_idx, tag_idx, mask)
            #     loss = -loss


            n += step
            n_batch += 1
            ner_model.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ner_model.eval()
                if n_batch % 40 == 0:
                    print(f"epoch: {j}, batch: {n_batch},loss: {loss}")
                    print("lr: ", optimizer.param_groups[0]['lr'])
                    if model_used == 'lstm':
                        tagScore = ner_model(dev_sentence_idx)
                        _, predictId = torch.max(tagScore, dim=-1)
                    elif model_used in ['lstm_crf','bert_lstm_crf','bert_crf']:
                        all_predictId = []
                        step_dev = step*10
                        n_dev = 0
                        while n_dev < dev_sentence_idx.shape[0]:
                            _,emission = ner_model(dev_sentence_idx[n_dev:n_dev+step_dev,], dev_tag_idx[n_dev:n_dev+step_dev,], dev_mask[n_dev:n_dev+step_dev,])
                            # print(emission.shape)
                            predictId = ner_model.decode(emission)
                            all_predictId += predictId
                            n_dev += step_dev

                        predictId = all_predictId
                    if type(predictId) is not list:
                        predictId = predictId.tolist()

                    # assert all(['B-[CLS]' not in one and 'B-[SEP]' not in one and 'B-PAD' not in one for one in trueTag])
                    predictTag_filter = remove_special_token_in_tags_for_predicted(predictId,dev_mask.tolist(),idx2tag,model_used)
                    # assert all(['B-[CLS]' not in one and 'B-[SEP]' not in one and 'B-PAD' not in one for one in predictTag_filter])


                    f1score = f1_score(trueTag,predictTag_filter)
                    if f1score > max_f1score:
                        max_f1score = f1score
                        with open(f'../model/{model_used}_{data_used}_f1.txt','w') as f:
                            f.write(str(max_f1score))
                        torch.save(ner_model.state_dict(), f'../model/{model_used}_{data_used}.pkl')
                        print(classification_report(trueTag, predictTag_filter))
                        trueTag2 = [tag for taglist in trueTag for tag in taglist]
                        predictTag2 = [tag for taglist in predictTag_filter for tag in taglist]
                        evaluate(trueTag2, predictTag2, verbose=True)
                ner_model.train()

def test():
    model_used = 'bert_lstm_crf'
    data_used = "ConLL2003"
    trained_model_path = f'../model/final/{model_used}_{data_used}.pkl'
    bert_path = '/home/share/Models/bert-large-cased/'
    word2idx = json.load(open(f'../map/{data_used}/word2idx.json', 'r'))
    tag2idx = json.load(open(f'../map/{data_used}/tag2idx.json', 'r'))
    ner_model, tag2idx, idx2tag = build_model(model_used, word2idx, tag2idx, bert_path)
    ner_model.load_state_dict(torch.load(trained_model_path))
    # print(ner_model)
    # exit()
    test_data_path = f'../data/{data_used}/test.txt'
    test_dataloader = ConLL_dataloader(test_data_path)
    test_sent = test_dataloader.sentences
    test_tag = test_dataloader.tags
    if model_used in ['lstm', 'lstm_crf']:
        trueTag = copy.deepcopy(test_tag)
    test_sentence_idx, test_tag_idx, test_mask = build_data(test_sent, test_tag, model_used, test_dataloader, tag2idx,
                                                         word2idx)
    test_sentence_idx = tensor(test_sentence_idx, requires_grad=False).to('cuda')
    test_tag_idx = tensor(test_tag_idx, requires_grad=False).to('cuda')
    test_mask = tensor(test_mask, requires_grad=False).to('cuda')
    assert test_sentence_idx.shape == test_tag_idx.shape == test_mask.shape
    if model_used in ['bert_lstm_crf','bert_crf']:
        trueTag = remove_special_token_in_tags_for_true(test_tag_idx.tolist(), test_mask.tolist(), idx2tag, tag2idx,
                                                        model_used)
    # print(dev_sentence_idx.shape)
    ner_model.to('cuda')
    ner_model.eval()
    with torch.no_grad():
        if model_used == 'lstm':
            tagScore = ner_model(test_sentence_idx)
            _, predictId = torch.max(tagScore, dim=-1)
        elif model_used in ['lstm_crf', 'bert_lstm_crf','bert_crf']:
            all_predictId = []
            step_test = 320
            n_test = 0
            while n_test < test_sentence_idx.shape[0]:
                _, emission = ner_model(test_sentence_idx[n_test:n_test + step_test, ], test_tag_idx[n_test:n_test + step_test, ],
                                        test_mask[n_test:n_test + step_test, ])
                # print(emission.shape)
                predictId = ner_model.decode(emission)
                all_predictId += predictId
                n_test += step_test

            predictId = all_predictId
        if type(predictId) is not list:
            predictId = predictId.tolist()
        predictTag_filter = remove_special_token_in_tags_for_predicted(predictId, test_mask.tolist(), idx2tag, model_used)

        f1score = f1_score(trueTag, predictTag_filter)
        print(classification_report(trueTag, predictTag_filter))
        trueTag2 = [tag for taglist in trueTag for tag in taglist]
        predictTag2 = [tag for taglist in predictTag_filter for tag in taglist]
        evaluate(trueTag2, predictTag2, verbose=True)



if __name__ == '__main__':
    # train()
    test()
