import numpy as np
import torch
import subprocess
import copy
import argparse

from tqdm import tqdm, trange

from nltk.tokenize import sent_tokenize
from transformers import(AutoTokenizer,AutoConfig, AutoModelForSequenceClassification )
from model_train.model import RobertaForSequenceClassification
from model_train.bert_model import BertForSequenceClassification


from common.utils_plus import *
from process_data.process_test import *
from sklearn.metrics.pairwise import cosine_similarity

SR = "Roberta-SR"
SMIA = "Roberta-SMIA"
SR_PARA = "Roberta-PARASR"
SR_PARA_BERT = "Bert-PARASR"


model_num_label = {
        SR:1,
        SMIA:1,
        SR_PARA:1,
        SR_PARA_BERT:1
    }

dev = {
    "hotpot": "dataset/raw/hotpot/hotpot_dev_distractor_v1.json",
}

processor = {
    'hotpot':hotpot_para
}

def load_model_tokenizer(model_paths):
    
    cuda_device_idx = 0
    cuda_devices = [0]
    assert len(cuda_devices) <= torch.cuda.device_count()
    CUDA_DEVICE_COUNT = len(cuda_devices) #torch.cuda.device_count()

    models, tokenizers = {}, {}
    device_mapping = {}
    for k, model_source in model_paths.items():
        
        config = AutoConfig.from_pretrained(model_source, num_labels=model_num_label[k])
        
        print("==", k, "==")
        print("load model from ", model_source)
        if "checkpoint" in model_source:
            idx = model_source.index("checkpoint")
            tokenizer_model_source = model_source[:idx]
        else:
            tokenizer_model_source = model_source
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_source)
        
        # tokenizer.add_tokens("<key>")
        # tokenizer.add_tokens("</key>")
        if k == SR_PARA:
            model = RobertaForSequenceClassification.from_pretrained(model_source, config=config)
        elif k == SR_PARA_BERT:
            model = BertForSequenceClassification.from_pretrained(model_source, config=config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_source, config=config)
        # model.resize_token_embeddings()

        device = torch.device(f"cuda:{cuda_devices[cuda_device_idx % CUDA_DEVICE_COUNT]}")

        model.to(device)

        models[k] = model
        tokenizers[k] = tokenizer

        device_mapping[k] = device
        cuda_device_idx += 1
        print(f'Loaded model and tokenizer for {k}.')

    return models, tokenizers, device_mapping


def tokenize(query, sentences, tokenizer, max_length=128, DEFAULT_LABEL=0):
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels = [], [], [], []
    if sentences is None:
        for i, s in enumerate(query):
            tok = tokenizer.encode_plus(text=s, max_length=max_length, padding='max_length', truncation=True)
            all_input_ids.append(tok['input_ids'])
            
            all_attention_mask.append(tok['attention_mask'])
            if 'token_type_ids' in tok:
                all_token_type_ids.append(tok['token_type_ids']) 
            all_labels.append(DEFAULT_LABEL)

    else:
        for i, s in enumerate(sentences):
            if isinstance(query, str):
                tok = tokenizer.encode_plus(text=query, text_pair=s, max_length=max_length, padding='max_length', truncation=True)
            else:
                tok = tokenizer.encode_plus(text=query[i], text_pair=s, max_length=max_length, padding='max_length', truncation=True)
            all_input_ids.append(tok['input_ids'])
            all_attention_mask.append(tok['attention_mask'])
            if 'token_type_ids' in tok:
                all_token_type_ids.append(tok['token_type_ids']) 
            all_labels.append(DEFAULT_LABEL)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    return [all_input_ids, all_attention_mask, all_token_type_ids, all_labels]


def mmr( sent_embeddings, keywords_idx, diversity):

    # Extract similarity within words, and between words and the document
    sent_embeddings = sent_embeddings.detach().cpu()
    word_similarity = cosine_similarity(sent_embeddings)
    
    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [keywords_idx]
    candidates_idx = [i for i in range(len(sent_embeddings)) if i != keywords_idx[0]]
    
    for _ in range(len(sent_embeddings) - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        # print(target_similarities)
        # Calculate MMR
        mmr = target_similarities.reshape(-1, 1)
       
        min_idx = np.argmin(mmr)
        min_sim = mmr[min_idx]
        if min_sim>diversity:
            
            return keywords_idx
        
        mmr_idx = candidates_idx[min_idx]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)
    return keywords_idx
    

def test_para(questions, models, tokenizers, device_mapping, cache): 
    
    save_0 = {'answer':{}, 'sp':{}}
    save_1 = {'answer':{}, 'sp':{}}
    neg_gap = []
    posi_gap = []
    for i, question in enumerate(tqdm(questions)) :
        q_id = question['_id']
        query = question['para']
        titles = question['titles']
        para_labels = question['para_labels']
        save_0['answer'][q_id] = ""
        save_0['sp'][q_id] =[]
        save_1['answer'][q_id] = ""
        save_1['sp'][q_id] =[]
        with torch.no_grad():
            for name, model in models.items():
                device = device_mapping[name]
                tokenizer = tokenizers[name]
                if len(query) == 0:
                    continue
                batch = tokenize(query, None, tokenizer, max_length=512)
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if "Roberta" not in name:
                    inputs["token_type_ids"] = batch[2] 
                score = model.predict(**inputs) # score[0] para score, score[1] sents score
                
                # if it is sr models do softmax.
                
                if model_num_label[SR] >1 :
                    score_para = torch.max(score[0], dim=1)[1]
                else:
                    score_para = score['para_logits'].view(-1)
                    score_para = score_para.detach().cpu()
                    score_sents = score['sent_logits'].view(-1)
                    score_sents = score_sents.detach().cpu()
                    all_input_f = score['input_f']
                    sent_segment = score['sent_segment']
                combined_scores = []
                for para_idx, para_score in enumerate(score_para):
                    seg = torch.where(sent_segment==para_idx)[0]
                    sent_score = torch.max(score_sents[seg])
                    comb_s = sent_score.item() + para_score.item()
                    combined_scores.append(comb_s)
                    gap = abs(sent_score.item()-para_score.item())
                    if para_labels[para_idx] == 1:
                        posi_gap.append(gap)
                    else:
                        neg_gap.append(gap)  
                combined_scores = np.array(combined_scores)
                top_para = np.argsort(combined_scores)[-2:][::-1]
                top_para2 = torch.topk(score_para,2)[1]
                questions[i]['topk_sents'] =  []
                for para_idx in top_para:
                                        
                    seg = torch.where(sent_segment==para_idx)[0]
                    pred_sents = torch.where(score_sents[seg]>0)[0]
                    pred_sents = pred_sents.tolist()
                    
                    if len(pred_sents)>=1:
                        for sent_idx in pred_sents:
                            save_0['sp'][q_id].append([titles[para_idx], sent_idx]) 
                        
                    else:
                        
                        sent_idx = torch.topk(score_sents[seg],1)[1]
                        save_0['sp'][q_id].append([titles[para_idx], sent_idx.item()]) 
                
                for para_idx in top_para2:
                                        
                    seg = torch.where(sent_segment==para_idx)[0]
                    pred_sents = torch.where(score_sents[seg]>0)[0]
                    pred_sents = pred_sents.tolist()
                    
                    if len(pred_sents)>=1:
                        for sent_idx in pred_sents:
                            save_1['sp'][q_id].append([titles[para_idx], sent_idx]) 
                        
                    else:
                        
                        sent_idx = torch.topk(score_sents[seg],1)[1]
                        save_1['sp'][q_id].append([titles[para_idx], sent_idx.item()]) 
                        
                
    json_plus.dump(save_0, cache)
    json_plus.dump(save_1, cache.replace(".json", "_1.json"))
    gap = {
        'posi_gap':round(sum(posi_gap)/len(posi_gap), 2),
        'neg_gap':round(sum(neg_gap)/len(neg_gap), 2),
        'combined_gap': round(sum(neg_gap+posi_gap)/len(neg_gap+posi_gap), 2)
    }
    json_plus.dump(gap, cache.replace(".json", "_gap.json"))
    return 




def test(questions, models, tokenizers, device_mapping, cache): 
    
    pred = []
    gold = []
    top_one = []
    distribution = {}
    for i, question in enumerate(tqdm(questions)) :
        
        candidates = question['sents']
        query = question['question']
        
        with torch.no_grad():
            for name, model in models.items():
                device = device_mapping[name]
                tokenizer = tokenizers[name]
                batch = tokenize(query, candidates, tokenizer)
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if "Roberta" not in name:
                    inputs["token_type_ids"] = batch[2] 
                score = model(**inputs)
                
                # if it is sr models do softmax.
                
                if model_num_label[SR] >1 :
                    score = torch.max(score[0], dim=1)[1]
                else:
                    score = score[0].view(-1)
                
                top_results_init = torch.where(score > 0)                 
                questions[i]['topk_sents'] =  []
                for idx in top_results_init[0]:
                    
                    questions[i]['topk_sents'].append([score[idx].item(),idx.item()])
                    
                
               
    json_plus.dump(questions, cache)
    return 

def top1_two_hop(gold, pred, save_path, models, tokenizers, device_mapping): 
    
    save = {'answer':{}, 'sp':{}}
    for go, pr in tqdm(zip(gold,pred), total=len(gold)):
        assert go['question'] == pr['question'], "questions do not match"
        q_id = go['_id']
        pr_sp = pr['topk_sents']
        titles = pr['titles']
        all_sents = pr['sents']
        query = pr['question']
        
        pr_sp= sorted(pr_sp, key=lambda x: x[0], reverse=True)
        try:
            top1idx = pr_sp[0][1] # idx of top 1 in sents
            top1sent = all_sents[top1idx]
            top1title = titles[top1idx][0]
            top2title = ""
            other_sents = all_sents[:top1idx]+all_sents[top1idx+1:]
            save['sp'][q_id] =[titles[top1idx]]
        except:
            top1sent, top1title,top2title, ="","",""
            other_sents = all_sents
            save['sp'][q_id] =[]
        
        save['answer'][q_id] = ""
        
        
        with torch.no_grad():
            
            SMIA_model = models[SMIA]
            device = device_mapping[SMIA]
            tokenizer = tokenizers[SMIA]
            batch = tokenize(query+top1sent, other_sents, tokenizer)
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if "Roberta" not in SMIA:
                inputs["token_type_ids"] = batch[2] 
            score = SMIA_model(**inputs)
            if model_num_label[SMIA] >1 :
                score = torch.max(score[0], dim=1)[1]
            else:
                score = score[0].view(-1)
            sorts, indices = torch.sort(score, descending=True)
            for sorted_score, ind in zip(sorts, indices):
                if sorted_score < 0: # non missing sents
                    break
                miss_sent = all_sents[ind]
                miss_sent_idx = all_sents.index(miss_sent)
                miss_sent_title = titles[miss_sent_idx][0]
                if top1title == "":
                    top1title = miss_sent_title
                if top2title == "" and miss_sent_title!=top1title:
                    top2title = miss_sent_title
                if miss_sent_title in [top1title, top2title]:
                    save['sp'][q_id].append(titles[miss_sent_idx])
    json_plus.dump(save, save_path)
    print("done")

    return 

def top2_two_hop(questions, models, tokenizers, device_mapping): 
    
    cache_data = {'questions':[]}
    total = 0
    pred = []
    gold = []
    top_one_list = []
    for i, question in enumerate(tqdm(questions)) :
        
        candidates = question['sents']
        query = question['question']
        top_k = len(question['label'])
        with torch.no_grad():
                sr_model = models[SR]
                device = device_mapping[SR]
                tokenizer = tokenizers[SR]
                batch = tokenize(query, candidates, tokenizer)
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if "Roberta" not in SR:
                    inputs["token_type_ids"] = batch[2] 
                score1 = sr_model(**inputs)
                # if it is sr models do softmax.
                
                if model_num_label[SR] >1 :
                    score1 = torch.max(score1[0], dim=1)[1]
                else:
                    score1 = score1[0].view(-1)
                
                top_two_r1_score, top_two_r1 = torch.topk(score1, k=2)[0].tolist(), torch.topk(score1, k=2)[1].tolist()
                top_two_r2_score, top_two_r2 = [], []
                for top in top_two_r1:
                    topsent = candidates[top] 
                
                    candidates_2 = []
                    for c in candidates:
                        if c != topsent:
                            candidates_2.append(c)
                    
                    # second round
                    SMIA_model = models[SMIA]
                    device = device_mapping[SMIA]
                    tokenizer = tokenizers[SMIA]
                    batch = tokenize(query+topsent, candidates_2, tokenizer)
                    batch = tuple(t.to(device) for t in batch)
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    if "Roberta" not in SMIA:
                        inputs["token_type_ids"] = batch[2] 
                    score2 = SMIA_model(**inputs)
                    if model_num_label[SMIA] >1 :
                        score2 = torch.max(score2[0], dim=1)[1]
                    else:
                        score2 = score2[0].view(-1)
                    
                    r2_score, r2_index = torch.topk(score2, k=1)[0].item() , torch.topk(score2, k=1)[1].item() 
                    topsent = candidates_2[r2_index] 
                    r2_index = candidates.index(topsent)
                    top_two_r2_score.append(r2_score)
                    top_two_r2.append(r2_index)


                if top_two_r1_score[0] + top_two_r2_score[0] > top_two_r1_score[1] + top_two_r2_score[1]:
                    top_results = sorted([top_two_r1[0], top_two_r2[0]])
                else:
                    top_results = sorted([top_two_r1[1], top_two_r2[1]])
                pred.append(sorted(top_results))
                gold.append(sorted(question['label']))
                questions[i]['topk_sents'] =  [ [top_two_r1[0], top_two_r2[0]],[top_two_r1[1], top_two_r2[1]] ]
               
    json_plus.dump(questions, "cache_folder/sr_hotpot_bm25_1_con_smia.jsonl")              
    acc = accuracy(pred, gold)
    
    return acc
    

def temp():
    sr_hotpot = json_plus.load("eval_folder/para_hotpot_random.json")['sp']
    sr_hotpot_bm1 = json_plus.load("eval_folder/para_hotpot_random_mmr.json")['sp']
    
    for i, (pred, pred_bm1 ) in enumerate(tqdm(zip(sr_hotpot, sr_hotpot_bm1))):
        # if len(sr_hotpot_bm1[pred_bm1]) > 2:
            print("before mmr",sr_hotpot[pred])
            print("after mmr",sr_hotpot_bm1[pred_bm1])

       
   

def temp2(gold, pred, save_path):
    save = {'answer':{}, 'sp':{}}
    for go, pr in zip(gold, pred):
        assert go['question'] == pr['question'], "questions do not match"
        q_id = go['_id']
        pr_sp = pr['topk_sents']
        titles = pr['titles']
        save['answer'][q_id] = ""
        save['sp'][q_id] =[]
        
        pr_sp= sorted(pr_sp, key=lambda x: x[0], reverse=True)
        title = []
        for i in pr_sp:
            if len(title) <2:
                title.append(titles[i[1]][0])
            if titles[i[1]][0] in title:
                save['sp'][q_id].append(titles[i[1]])
    json_plus.dump(save, save_path)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = "/scratch/mluo26/sia_models/para_base/")
    parser.add_argument("--save_path", type=str, default = "eval_folder/hotpot/para_sent_con_triple.json")
    args = parser.parse_args()
    model_paths = {
        SR_PARA :args.model_path
    }
    models, tokenizers, device_mapping = load_model_tokenizer(model_paths)
    for name, path in dev.items():
        
        if name == "hotpot":
                
            gold = processor[name](path = path)
            print("save file in ", args.save_path)
            test_para(gold,models,tokenizers, device_mapping,args.save_path)