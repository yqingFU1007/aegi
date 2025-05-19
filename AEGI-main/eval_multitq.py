from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch
import json
from datetime import datetime
import re
from kg_processor import KG_Process
import ast
import openai
import difflib
import numpy as np

from transformers import DistilBertTokenizer, DistilBertModel
from dateutil.parser import parse

# from sentence_transformers import SentenceTransformer, util
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import RobertaTokenizer, RobertaForSequenceClassification

from prompts_best import deepseek_v3_multi


from prompts_best import prompt_tq_answer, prompt_tq_rerank, prompt_multi_rerank, prompt_multi_answer, get_multi_prompt

def is_valid_date(date_str):
    formats = ['%Y', '%Y-%m', '%Y-%m-%d']

    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue

    return False
    
def check_substring_multi(arr1, arr2):
    for str1 in arr1:
        for str2 in arr2:
            if str1==str2 or (is_valid_date(str2) and str(str2) in str(str1)):
                print(f'{str1} in {str2}')
                return True
    return False

def caculate_acc_multi(topk_answers, i, actual_answers, alignment_ans_jsn, hits_at_k, total):

    predicted = [topk_answers[i][0]]
         
    if alignment_ans_jsn.get(i) is not None:
        predicted = set(predicted) | set(alignment_ans_jsn.get(i))#合并生成答案 
    else:
        predicted = set(predicted)
    print(f'predicted     : {predicted}')
    print(f'actual_answers: {actual_answers}')
    if check_substring_multi(list(predicted), list(actual_answers)):
        hits_at_k += 1 
    total += 1
    eval_accuracy = hits_at_k / total
    print(f'correct: {hits_at_k}, total: {total}')
    print('Current acc Hit 1 :%f'%(round(eval_accuracy, 3)))
    if (total % 1000)==0:
        with open("/root/autodl-tmp/MultiTQ-main/MultiQA/results/MultiTQ/tmp_0426.txt", "a", encoding="utf-8") as file:
            file.write(f'\n Total: {total}, Acc: {round(eval_accuracy, 3)}')
    return hits_at_k, total

def get_list_from_string(text):
    if text is not None:
        pattern = r'(\[.*?\])'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                m = match.group(1).strip()
                list_result = ast.literal_eval(m)
                return list_result
            except SyntaxError:
                return []
            except ValueError:
                return []
    return []
def get_llm_answer(text):
    if text is not None:
        list_result = text.split(',')
        for i,t in enumerate(list_result):
            list_result[i] = t.strip()
        return list_result
    return []

def get_confidence_score(text):
    if text is not None:
        pattern = r'\d+'
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group())
            except SyntaxError:
                return 0
    return 0


def eval_multi_aegi(qa_model, dataset, batch_size=128, split='valid', k=10, target_acc=0.8, max_k=100):
    num_workers = 4
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k  # not change name in fn signature since named param used in places
    k_list = [1,3,5,10]
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)
    with open('/root/autodl-tmp/MultiTQ-main/data/MultiTQ/questions/processed_questions_ext_quad/test_anchor_0421.json','r',encoding='utf-8') as f:
        anchor = json.load(f)
    with open('/root/autodl-tmp/MultiTQ-main/data/MultiTQ/kg/entity2id.json','r',encoding='utf-8') as f:
        entity2id = json.load(f)
    

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    kg_process = KG_Process("/root/autodl-tmp/MultiTQ-main/data/MultiTQ/kg/full.txt")
    
    for i_batch, a in enumerate(loader):
        # if size of split is multiple of batch size, we need this
        # todo: is there a more elegant way?
        if i_batch * batch_size == len(dataset.data):
            break
        answers_khot = a[-1]  # last one assumed to be target
        scores = qa_model.forward(a)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        loss = qa_model.loss(scores, answers_khot.cuda())
        total_loss += loss.item()
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    alignment_ans_jsn = {}
    hits_at_k = 0
    total = 0
    
    for i, q in tqdm(enumerate(dataset.data),total=len(dataset.data)):
        print("-------------")
        print(f'Question: {q['question']}')
        entities_text = q['entities']
        if len(anchor[i]['anchor'])>0:
            anchor_event = anchor[i]['anchor'][0]
            rel = [anchor_event[1]]
        else:
            anchor_event = ""
            rel = [anchor[i]['rel']]
        print(f'anchor_event: {anchor_event}')
        condidate_answers = topk_answers[i][:15]
        q_facts, q_for_nodes = kg_process.extract_quad_from_anchor(condidate_answers,q['question'], entities_text,q['time'], rel, anchor_event)
        
        dataset.data[i]['alignment_ans'] = []

        for fac in q_facts:
            print(fac)
       
      
        try: 
            score_rerank = prompt_multi_rerank(condidate_answers, q['question'], q_facts, q_for_nodes,q['answer_type'])
            print(f"Score rerank: {score_rerank}")
            if score_rerank is None:
                rerank_ans = []
                confidence_score = 0
            elif len(score_rerank.rsplit('\n', 1)) > 1:
                rerank_ans = get_list_from_string(str(score_rerank.rsplit('\n', 1)[-2]))
                confidence_score = get_confidence_score(score_rerank.rsplit('\n', 1)[-1])
            else:
                rerank_ans = get_list_from_string(str(score_rerank.rsplit('\n', 1)[-1]))
                confidence_score = get_confidence_score(score_rerank.rsplit('\n', 1)[-1])
            print(f'Confidence score = {confidence_score}')
            if confidence_score < 7:
                print("Re-answer by llm!")  
                q['re_answer'] = True
                re_answer = prompt_multi_answer(q['question'], q_facts)
                # print(re_answer)
                last_line = re_answer.rsplit('\n', 1)[-1]
                # print(f'last_line = {last_line}')
                
                alignment_ans = []
                if last_line != "None":
                    if is_valid_date(last_line):
                        alignment_ans.append(last_line)
                    answer_list = get_llm_answer(last_line)
                    for a in answer_list:
                        m = difflib.get_close_matches(a, entity2id.keys(),n=1)
                        if len(m) > 0:
                            alignment_ans.append(m[0].replace("_", " "))
                        if is_valid_date(a):
                            alignment_ans.append(a)
                # else:
                #     re_answer = prompt_multi_answer(q['question'], anchor[i][])
                if len(alignment_ans)>0:
                    # print(f'alignment_ans = {alignment_ans}')
                    dataset.data[i]['alignment_ans'] = alignment_ans
                    alignment_ans_jsn[i] = dataset.data[i]['alignment_ans']
                    rerank_ans = alignment_ans + rerank_ans
                    # print(f'------re-answer list-------:{rerank_ans}')
        except openai.BadRequestError:
            print("Request string too long!")
            hits_at_k, total = caculate_acc_multi(topk_answers, i, q['answers'],alignment_ans_jsn, hits_at_k, total)
            continue
        if len(q['answers'])==0:
            hits_at_k, total = caculate_acc_multi(topk_answers, i, q['answers'],alignment_ans_jsn, hits_at_k, total)
            continue
        if len(rerank_ans)==0:
            hits_at_k, total = caculate_acc_multi(topk_answers, i, q['answers'],alignment_ans_jsn, hits_at_k, total)
            continue
        print(f'Ground truth: {q['answers']}')
        print(f'Original rank: {topk_answers[i][:15]}')
        if len(rerank_ans)>0:
            topk_answers[i] = rerank_ans
        print(f'Result: {rerank_ans}')
        hits_at_k, total = caculate_acc_multi(topk_answers, i, q['answers'],alignment_ans_jsn, hits_at_k, total)
    
    

    eval_accuracy_for_reporting = 0
    acc_list = []
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)
        time_level_count = defaultdict(list)
        time_level_count_all = defaultdict(list)
        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['qtype']
            time_level = question['time_level']
            if 'Single' in question['qlabel']:
                simple_complex_type = 'Single'
            else:
                simple_complex_type = 'Multiple'
            entity_time_type = question['answer_type']
            # question_type = question['template']

            
                        
            predicted = topk_answers[i][:k]
            if alignment_ans_jsn.get(i) is not None:
                predicted = set(predicted) | set(alignment_ans_jsn.get(i))#merge generation answer
            else:
                predicted = set(predicted)
            
            

            # multiple time answers - hard way
            if question['answer_type']=='time':
                if len(question['answers'][0]) == 4:
                    predicted = [x[:4] for x in predicted]
                elif len(question['answers'][0]) == 7:
                    predicted = [x[:7] for x in predicted]

            if len(set(actual_answers).intersection(set(predicted))) > 0:
                val_to_append = 1
                hits_at_k += 1
            else:
                if check_substring_multi(list(predicted), list(actual_answers)):
                    hits_at_k += 1 
                val_to_append = 0

            question_types_count[question_type].append(val_to_append)
            if (question['qtype'] == 'before_after' and len(question['time']) > 0) or (
                    question['qtype'] == 'equal') or (question['qtype'] == 'equal_multi'):
                time_level_count_all[time_level].append(val_to_append)
            if (question['qtype'] == 'before_after' and len(question['time']) > 0) or (
                    question['qtype'] == 'equal') or (question['qtype'] == 'equal_multi'):
                time_level_count[question['qtype'] + '-' + time_level].append(val_to_append)
            simple_complex_count[simple_complex_type].append(val_to_append)
            entity_time_count[entity_time_type].append(val_to_append)
            total += 1

        eval_accuracy = hits_at_k / total
        acc_list.append(eval_accuracy)
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))

        time_level_count_all = dict(sorted(time_level_count_all.items(), key=lambda x: x[0].lower()))
        time_level_count = dict(sorted(time_level_count.items(), key=lambda x: x[0].lower()))
        question_types_count = dict(sorted(question_types_count.items(), key=lambda x: x[0].lower()))
        simple_complex_count = dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower()))
        entity_time_count = dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower()))
        # for dictionary in [question_types_count]:
        for dictionary in [question_types_count, simple_complex_count, time_level_count_all, time_level_count,
                           entity_time_count]:
            # for dictionary in [simple_complex_count, entity_time_count]:
            for key, value in dictionary.items():
                hits_at_k = sum(value) / len(value)
                s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
                    q_type=key,
                    hits_at_k=round(hits_at_k, 3),
                    num_questions=len(value)
                )
                if print_numbers_only:
                    s = str(round(hits_at_k, 3))
                eval_log.append(s)
            eval_log.append('')
    formatted_list = [f'{x:.4f}' for x in acc_list]
    selected_k = -1
    for i,x in enumerate(acc_list):
        if x > target_acc:
            selected_k = i+1
            break
    print(f'acc_list : {formatted_list}')
    # with open('answer.json', 'w',encoding='utf-8') as obj:
    #     obj.write(json.dumps(dataset.data, indent=4,ensure_ascii=False))
    
    
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log, selected_k
