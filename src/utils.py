import json
import os

def generate_word2index_tag2index(data_dir,outdir,sep='\t'):
    fileList = os.listdir(data_dir)
    wordList = []
    tagList = []
    n = 0
    wordList.append('[PAD]')
    tagList.append('B-PAD')
    for file in fileList:
        with open(data_dir+'/'+file,'r',encoding='utf-8') as f:
            for line in f:
                if line != '\n':
                    line = line.strip().split(sep)
                    wordList.append(line[0])
                    tagList.append(line[-1])

    wordList = sorted(list(set(wordList)))
    tagList = sorted(list(set(tagList)))
    idx2word = dict(enumerate(wordList))
    idx2tag = dict(enumerate(tagList))
    word2idx = {word:idx for idx,word in idx2word.items()}
    tag2idx = {tag:idx for idx,tag in idx2tag.items()}
    json.dump(word2idx, open(outdir+'/word2idx.json', 'w'))
    json.dump(tag2idx, open(outdir+'/tag2idx.json', 'w'))




if __name__ == "__main__":
    generate_word2index_tag2index('../data/AGAC', '../map/AGAC','\t')