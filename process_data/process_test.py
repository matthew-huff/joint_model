import json
from tqdm import tqdm 
import os

from common.utils_plus import *
from common.search import pyserini_bm25
import math

def hotpot_para(path):
    res =[]
    data = json_plus.load(path)
    for item in tqdm(data):
        question = item['question'] 
        answer = item['answer']
        qid = item['_id']
        all_paras =[]
        titles = []
        para_labels = []
        sent_labels = []
        
        supporting_title = {}
        for sp in item['supporting_facts']:
            title = sp[0]
            idx = sp[1]
            if title not in supporting_title:
                supporting_title[title] = [idx]
            else:
                supporting_title[title].append(idx)
        
        for para in item['context']:
            title = para[0]
            titles.append(title)
            all_paras.append("</s> ".join([question]+para[1][:9]))
            sent_label = [0]*len(para[1][:9])
            
            if title in supporting_title:
                para_labels.append(1)
                for idx in supporting_title[title]:
                    if idx < len(sent_label):
                        sent_label[idx] = 1
            else:
                para_labels.append(0)
            sent_labels.append(sent_label)
                
        datum = {
            '_id': qid,
            'para':all_paras,
            'answer':answer,
            'titles':titles,
            'para_labels':para_labels,
            'sent_labels':sent_labels     
        }
        res.append(datum)
    return res


def hotpot(path):
    res = []
    data = json_plus.load(path)
    distribution = {}
    idx_missing = 0
    total = 0
    for item in data:
        question = item['question'] 
        answer = item['answer']
        qid = item['_id']
        
        supporting_facts = item['supporting_facts']
        if (len(supporting_facts)) not in distribution:
            distribution[len(supporting_facts)] =1
        else:
            distribution[len(supporting_facts)] += 1
        supporting_title = {}
        for sp in item['supporting_facts']:
            title = sp[0]
            idx = sp[1]
            if sp[0] not in supporting_title:
                supporting_title[title] = [idx]
            else:
                supporting_title[title].append(idx)
        total += len(supporting_title)
            
        all_sents = []
        label = []
        titles = []
        for sents in item['context']:
            title = sents[0]
            if  title in supporting_title:
                for idx in supporting_title[title]:
                    if idx < len(sents[1]):
                        label.append(len(all_sents)+idx)  
                    else:
                         idx_missing += 1  
                
            all_sents.extend(sents[1])
            
            for i in range(len(sents[1])):
                titles.append([title, i])

        res.append({
            '_id':qid,
            'question':question,
            'answer':answer,
            'label':label,
            'sents':all_sents,
            'titles':titles
        })
    print("missing %.4f "%(idx_missing/total))
    return res


