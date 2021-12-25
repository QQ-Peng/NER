import torch.nn as nn
import torch.nn.functional as F
import torch
from torchcrf import CRF # pip install pytorch-crf,   version: 0.7.2
from transformers import BertModel

class BERT_Model(nn.Module):
    def __init__(self):
        super(BERT_Model, self).__init__()


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, tag_size, n_layer=1, reduction='mean',bidirection=True,pretraining_embed=False):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.num_direction = 1 if not bidirection else 2
        self.reduction = reduction
        self.embed = nn.Embedding(vocab_size, embed_dim) if not pretraining_embed else None
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=self.n_layer, batch_first=True, dropout=0.2, bidirectional=bidirection)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size) if reduction is not None else nn.Linear(hidden_dim*2, tag_size)

    def forward(self, sentences, pretraining_embed=None):
        embedding = self.embed(sentences) if pretraining_embed is None else pretraining_embed
        batch_size = sentences.shape[0]
        seq_len = sentences.shape[1]
        hidden_state = torch.randn(self.n_layer * self.num_direction, batch_size, self.hidden_dim).to(sentences.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(self.n_layer * self.num_direction, batch_size, self.hidden_dim).to(sentences.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, hidden = self.lstm(embedding,(hidden_state, cell_state))
        if self.reduction is not None:
            output = output.view(batch_size, seq_len, 2, -1).mean(dim=-2)
        # print(output.shape)
        tagSpace = self.hidden2tag(output)
        result = F.log_softmax(tagSpace, dim=2)
        return result

class LSTM_CRF_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, tag_size, n_layer=1, reduction='mean', bidirection=True,
                 pretraining_embed=False):
        super(LSTM_CRF_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.num_direction = 1 if not bidirection else 2
        self.reduction = reduction
        self.embed = nn.Embedding(vocab_size, embed_dim) if not pretraining_embed else None
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=self.n_layer, batch_first=True,
                            dropout=0.2, bidirectional=bidirection)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size) if reduction is not None else nn.Linear(hidden_dim * 2,
                                                                                                  tag_size)
        self.CRF = CRF(tag_size, batch_first=True)

    def forward(self, sentences, tags, mask, pretraining_embed=None):
        embedding = self.embed(sentences) if pretraining_embed is None else pretraining_embed
        batch_size = sentences.shape[0]
        seq_len = sentences.shape[1]
        hidden_state = torch.randn(self.n_layer * self.num_direction, batch_size, self.hidden_dim).to(
            sentences.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(self.n_layer * self.num_direction, batch_size, self.hidden_dim).to(
            sentences.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, hidden = self.lstm(embedding, (hidden_state, cell_state))
        if self.reduction is not None:
            output = output.view(batch_size, seq_len, 2, -1).mean(dim=-2)
        # print(output.shape)
        tagSpace = self.hidden2tag(output)
        emission = F.log_softmax(tagSpace, dim=2)
        # scalar
        log_likelihood = self.CRF(emission, tags, mask=mask.byte()).mean()

        return log_likelihood, emission

    def decode(self, emission):
        pred_tags = self.CRF.decode(emission)
        return pred_tags


class BERT_LSTM_CRF_Model(nn.Module):
    def __init__(self, lstm_hidden_dim, tag_size, n_layer=1, reduction='mean', bidirection=True,bert_path='/home/share/Models/bert-large-cased/'):
        super(BERT_LSTM_CRF_Model, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_layer = n_layer
        self.num_direction = 1 if not bidirection else 2
        self.reduction = reduction
        self.embed = BertModel.from_pretrained(bert_path)
        self.embed_dim = self.embed.embeddings.word_embeddings.embedding_dim
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=lstm_hidden_dim, num_layers=self.n_layer, batch_first=True,
                            dropout=0.3, bidirectional=bidirection)
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tag_size) if reduction is not None else nn.Linear(lstm_hidden_dim * 2,
                                                                                                  tag_size)
        self.CRF = CRF(tag_size, batch_first=True)

    def forward(self, sentences, tags, mask, pretraining_embed=None):
        embedding = self.embed(input_ids=sentences, attention_mask=mask)[0] if pretraining_embed is None else pretraining_embed
        batch_size = sentences.shape[0]
        seq_len = sentences.shape[1]
        hidden_state = torch.randn(self.n_layer * self.num_direction, batch_size, self.lstm_hidden_dim).to(
            sentences.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(self.n_layer * self.num_direction, batch_size, self.lstm_hidden_dim).to(
            sentences.device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, hidden = self.lstm(embedding, (hidden_state, cell_state))
        if self.reduction is not None:
            output = output.view(batch_size, seq_len, 2, -1).mean(dim=-2)
        # print(output.shape)
        tagSpace = self.hidden2tag(output)
        emission = F.log_softmax(tagSpace, dim=2)
        # scalar
        log_likelihood = self.CRF(emission, tags, mask=mask.byte()).mean()

        return log_likelihood, emission

    def decode(self, emission):
        pred_tags = self.CRF.decode(emission)
        return pred_tags



class BERT_CRF_Model(nn.Module):
    def __init__(self, tag_size, bert_path='/home/share/Models/bert-large-cased/'):
        super(BERT_CRF_Model, self).__init__()

        self.embed = BertModel.from_pretrained(bert_path)
        self.embed_dim = self.embed.embeddings.word_embeddings.embedding_dim
        self.hidden2tag = nn.Linear(self.embed_dim, tag_size)
        self.CRF = CRF(tag_size, batch_first=True)

    def forward(self, sentences, tags, mask, pretraining_embed=None):
        embedding = self.embed(input_ids=sentences, attention_mask=mask)[0] if pretraining_embed is None else pretraining_embed
        # print(output.shape)
        tagSpace = self.hidden2tag(embedding)
        emission = F.log_softmax(tagSpace, dim=2)
        # scalar
        log_likelihood = self.CRF(emission, tags, mask=mask.byte()).mean()

        return log_likelihood, emission

    def decode(self, emission):
        pred_tags = self.CRF.decode(emission)
        return pred_tags


