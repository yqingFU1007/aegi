import difflib
import json
import torch
import numpy as np
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from tqdm.contrib import tenumerate
from kg_processor import KG_Process
import re
from dateutil.parser import parse
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

from transformers import DistilBertTokenizer, DistilBertModel
from dateutil.parser import parse
from utils_llm import gpt_4o_mini_extract_tq_quant
from datetime import datetime

from prompts import extract_multitq_quadruples, prompt_multiqa_rewrite, extract_multitq_triples, gpt_4o_mini_multiqa


def extract_time(sentence: str) -> str:
    date_pattern = r'\b(?:\d{1,2}\s)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s?\d{1,2}?(?:\s|,)?\s?\d{2,4}\b|\b\d{4}\b'
    date_match = re.search(date_pattern, sentence)

    if date_match:
        date_str = date_match.group()
        date_obj = parse(date_str)
        date_format = ''

        if len(date_str.split(' ')) == 3:
            date_format = f"{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}"
        elif len(date_str.split(' ')) == 2:
            date_format = f"{date_obj.year}-{date_obj.month:02d}"
        elif len(date_str.split(' ')) == 1:
            date_format = f"{date_obj.year}"

        return [date_format]
    else:
        return []
    
def format_quadruple_multitq(quad):
    if len(quad)==4:
        head, relation, tail, time = quad
        return f"{head} {relation} {tail} on {time}."

def convert_float32(data):
    if isinstance(data, np.float32):
        return float(data)
    if isinstance(data, list):
        return [convert_float32(item) for item in data]
    if isinstance(data, dict):
        return {key: convert_float32(value) for key, value in data.items()}
    return data

def encode_text(text, tokenizer, model):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def sort_by_date(data):
    return sorted(data, key=lambda x: datetime.strptime(x[3], '%Y-%m-%d'))

def get_facts_with_time(t, times):
    return f'For fact ({t}), it happens on {times}'

def format_truple_multitq(quad):
    if len(quad)==3:
        head, relation, tail = quad
        return f"{head} {relation} {tail}."


def get_anchor_events(sim_model,kg_process, question,entities_text, time_list, rel):
    q_facts = list(kg_process.pre_extract_quads(entities_text,time_list,rel,'day'))
    if len(q_facts) == 0:
        return [],[],[], ''
    truples = set([])
    times = defaultdict(set)
    for quad in q_facts:
        head, rel, tail, time = quad
        tr = tuple([head,rel,tail])
        truples.add(tr)
        times[tr].add(time)

    question_rewrite = question
    print(f'q rewrite: {question_rewrite}')
    formatted_trs = [format_truple_multitq(truple) for truple in truples]
    question_embedding = sim_model.encode([question_rewrite]).reshape(1, -1)
    # quad_embeddings = [sim_model.encode(truple) for truple in formatted_trs]
    quad_embeddings = sim_model.encode(formatted_trs, batch_size=2048)
    similarities = util.cos_sim(quad_embeddings, question_embedding).flatten()
    sorted_pairs = sorted(zip(truples, similarities), key=lambda x: x[1], reverse=True)
    sorted_quads = [pair[0] for pair in sorted_pairs]  

    q_filter_truples_times = []
    q_re_quads = []
    for t in sorted_quads[:5]:
        head,rel,tail = t
        time_points = times[t]
        # time_points = sort_date(time_points, q['time_level'])
        facts_with_time = get_facts_with_time(t, time_points)
        # print(facts_with_time)
        q_filter_truples_times.append(facts_with_time)
        for time in times[t]:
            q_re_quads.append(tuple([head,rel,tail,time]))

    anchor_event = q_re_quads[0]
    score_dic = {}
    anchor_event = q_re_quads[0]
    print(f'anchor_event: {anchor_event}')
    em_emb = sim_model.encode(anchor_event, batch_size=2048)
    length = torch.tensor([len(em) for em in anchor_event])
    sim = util.cos_sim(em_emb, question_embedding).flatten()/length*len(question)
    print(f'sim: {sim}')
    for i, node in enumerate(anchor_event):
        score_dic[node] = sim[i].item()
    anchor_nodes = [
        node for node, score in zip(list(anchor_event), sim) 
        if score > 1.0
    ]
    anchor_nodes_else = [
        node for node, score in zip(list(anchor_event), sim) 
        if score <= 1.0
    ]
    facts_two_hop = kg_process.pre_extract_two_hop(question_rewrite, time, rel, anchor_nodes_else)
   
    return sorted_quads[:5],q_re_quads, facts_two_hop, anchor_event
    
    

if __name__ == '__main__':
    ent_path = '../data/MultiTQ/kg/tkbc_processed_data/ent_id'
    input_path = '../data/MultiTQ/questions/'
    output_path = '../data/MultiTQ/questions/processed_questions_ext_quad/'
    rel2id_path = '../data/MultiTQ/kg/relation2id.json'
    with open(ent_path,'r',encoding='utf-8') as f:
        ent_data = f.readlines()
    with open(rel2id_path,'r',encoding='utf-8') as f:
        rel2id = json.load(f)
    
    all_entities = []
    for i in ent_data:
        all_entities.append(i.strip().split('\t')[0])
    keys = all_entities
    tagger = SequenceTagger.load("flair/ner-english-large")
    kg_process = KG_Process("/root/autodl-tmp/MultiTQ-main/data/MultiTQ/kg/full.txt")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    rel_embs = [model.encode(r, batch_size=2048) for r in tqdm(rel2id.keys())]

   
    for d in ['dev']:
        correct_ans = 0
        print(d.upper() + ' DATASET PROCESSING......')
        with open(input_path+d+'.json','r',encoding='utf-8') as f:
            dataset = json.load(f)
        
        questions = [Sentence(x['question']) for x in dataset]
        for ix,s in tqdm(tenumerate(questions)):
            tagger.predict(s)
            e = []
            res = dataset[ix]['question']
            for entity in s.get_spans('ner'):
                print(entity.text)
                res = res.replace(entity.text, "")
                entity_text = difflib.get_close_matches(entity.text,keys,n=1)
                e.append({'entity':entity_text,'position':[entity.start_position,entity.end_position]})
                clean_ner_result = [y for y in e if len(y['entity'])>0]
            dataset[ix]['time'] = extract_time(dataset[ix]['question'])
            dataset[ix]['entities'] = [x['entity'][0] for x in clean_ner_result]

            res = res.replace("which", "").replace("Which", "").replace("who", "").replace("Who", "").replace("what", "").replace("What", "").replace("first", "").replace("last", "").replace("when", "").replace("When", "").replace("did", "").replace("with", "").replace("With", "").replace("from", "").replace("before", "").replace("after", "").replace("was", "").replace("7","").replace("8","").replace("9","").replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").replace("0","").replace(" ","_")
            question_embedding = model.encode([res]).reshape(1, -1)
            similarities = util.cos_sim(rel_embs, question_embedding).flatten()
            sorted_pairs = sorted(zip(rel2id, similarities), key=lambda x: x[1], reverse=True)
            sorted_rel = [pair[0] for pair in sorted_pairs] 
            sorted_scores = [pair[1] for pair in sorted_pairs]
            print(f'Question-res: {res}')
            for rel, score in sorted_pairs[:1]:
                print(f"Score: {score:.4f} | Rel: {rel}")
            dataset[ix]['rel'] = sorted_rel[0].replace("_", " ")
            dataset[ix]['score'] = float(sorted_scores[0])
            anchor_triples,quads, facts_two_hop, anchor_event = get_anchor_events(model,kg_process, dataset[ix]['question'],dataset[ix]['entities'], dataset[ix]['time'], sorted_rel[:1])
            for q in quads:
                print(q)
            dataset[ix]['anchors'] = quads
            dataset[ix]['anchor_triple'] = anchor_triples
            dataset[ix]['anchor_event'] = anchor_event
        with open(output_path + d + '_anchor.json', 'w',encoding='utf-8') as obj:
            obj.write(json.dumps(dataset[:ix+1], indent=4,ensure_ascii=False))
        print(d.upper() + ' DATASET PROCESSED')



