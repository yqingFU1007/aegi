from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import math


class KG_Process():
    def __init__(self, path):
        self.kg_quads = self.read_txt_quadruples_space(path)
        
        self.head2quad = defaultdict(set)
        self.tail2quad = defaultdict(set)
        self.head2quad_tm = defaultdict(set)
        self.tail2quad_tm = defaultdict(set)
        self.head2quad_ty = defaultdict(set)
        self.tail2quad_ty = defaultdict(set)
        self.time2quad = defaultdict(set)
        self.rel2quad = defaultdict(set)
        self.rel2quad_tm = defaultdict(set)
        self.rel2quad_ty = defaultdict(set)
        self.build_kg()
        self.index = None

    def read_txt_quadruples_space(self,file_path):
        quadruples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    elements = line.split()
                    if len(elements) == 4:
                        quadruples.append(tuple(elements))
        except FileNotFoundError:
            print(f"File {file_path} doesn't exist.")
        return quadruples


    def build_kg(self):
        print("Begin to build the knowledge graph...")
        for head, rel, tail, ts in tqdm(self.kg_quads):
            
            quad = []
            quad.append(head.replace("_", " "))
            quad.append(rel.replace("_", " "))
            quad.append(tail.replace("_", " "))
            quad.append(ts)

            quad_tm = []
            quad_tm.append(head.replace("_", " "))
            quad_tm.append(rel.replace("_", " "))
            quad_tm.append(tail.replace("_", " "))
            quad_tm.append(ts[:7])
            quad_ty = []
            quad_ty.append(head.replace("_", " "))
            quad_ty.append(rel.replace("_", " "))
            quad_ty.append(tail.replace("_", " "))
            quad_ty.append(ts[:4])
            
            
            self.head2quad[head.replace("_", " ")].add(tuple(quad))
            self.head2quad_tm[head.replace("_", " ")].add(tuple(quad_tm))
            self.head2quad_ty[head.replace("_", " ")].add(tuple(quad_ty))
            self.tail2quad[tail.replace("_", " ")].add(tuple(quad))
            self.tail2quad_tm[tail.replace("_", " ")].add(tuple(quad_tm))
            self.tail2quad_ty[tail.replace("_", " ")].add(tuple(quad_ty))
            self.rel2quad[rel.replace("_", " ")].add(tuple(quad))
            self.rel2quad_tm[rel.replace("_", " ")].add(tuple(quad_tm))
            self.rel2quad_ty[rel.replace("_", " ")].add(tuple(quad_ty))
            self.time2quad[ts].add(tuple(quad))
            self.time2quad[ts[:7]].add(tuple(quad_tm))
            self.time2quad[ts[:4]].add(tuple(quad_ty))
            
            

    def sort_by_date(self, data, time_level, ascending=True):
        # 按日期对数据进行排序
        if time_level == 'day':
            return sorted(data, key=lambda x: datetime.strptime(x[3], '%Y-%m-%d'), reverse=not ascending)
        elif time_level == 'month':
            return sorted(data, key=lambda x: datetime.strptime(x[3], '%Y-%m'), reverse=not ascending)
        else:
            return sorted(data, key=lambda x: datetime.strptime(x[3], '%Y'), reverse=not ascending)
    
        
    def extract_kg_quads(self, nodes, question_entities,time, rel, time_level):
        if len(nodes)==0:
            print("No candidate answers!")
            return []

        if time_level == "month":
            head2quad = self.head2quad_tm
            tail2quad = self.tail2quad_tm
            rel2quad = self.rel2quad_tm
        elif time_level == "year":
            head2quad = self.head2quad_ty
            tail2quad = self.tail2quad_ty
            rel2quad = self.rel2quad_ty
        else:
            head2quad = self.head2quad
            tail2quad = self.tail2quad
            rel2quad = self.rel2quad
        print(f'question_entities: {question_entities}, time: {time}')
        print(f'rel : {rel}')
        subset_e = set([])
        for i,e in enumerate(question_entities):
            hq = head2quad.get(e)
            tq = tail2quad.get(e)
            q = (set(hq) if hq is not None else set([])) | (set(tq) if tq is not None else set([]))
            subset_e = subset_e | set(q) if q is not None else set([])
            if len(rel) > 0 and len(time) == 0:
                r = rel2quad.get(rel[0])
                if r is not None:
                    subset_e = subset_e.intersection(r)
            
        subset = set([])
        if len(time) > 0:
            for t in time:
                for quad in subset_e:
                    if str(t) in str(quad[-1]):
                        subset.add(tuple(quad))
        else:
            subset = subset_e
        filter_q = defaultdict(set)
        for node in nodes:
            h = head2quad.get(node)
            t = tail2quad.get(node)
            ts = self.time2quad.get(node)
            
            relactive_quad = (set(h) if h is not None else set([])) | (set(t) if t is not None else set([])) | (set(ts) if ts is not None else set([]))
            if relactive_quad is not None and subset is not None:
                question_rel = relactive_quad.intersection(subset)
                filter_q[node] = question_rel
        return filter_q, subset

    # def pre_extract_two_hop(self, question, time, rel, anchor_nodes):
    #     question = question.lower()
    #     print(f'anchor_nodes: {anchor_nodes}, time: {time}, rel: {rel}')
    #     _time = []
    #     if 'month' in question:
    #         time_level = 'month'
    #         head2quad = self.head2quad_tm
    #         tail2quad = self.tail2quad_tm
    #         rel2quad = self.rel2quad_tm
    #         _time = []
    #         for t in time:
    #             _time.append(t[:7])
    #     elif 'year' in question:
    #         time_level = 'year'
    #         head2quad = self.head2quad_ty
    #         tail2quad = self.tail2quad_ty
    #         rel2quad = self.rel2quad_ty
    #         for t in time:
    #             _time.append(t[:4])
    #     else:
    #         time_level = 'day'
    #         head2quad = self.head2quad
    #         tail2quad = self.tail2quad
    #         rel2quad = self.rel2quad
    #         _time = time
    #     print(f'time_level: {time_level}')
    #     subset_e = set([])
    #     for e in anchor_nodes:
    #         hq = head2quad.get(e)
    #         tq = tail2quad.get(e)
    #         q = (set(hq) if hq is not None else set([])) | (set(tq) if tq is not None else set([]))
    #         subset_e = subset_e | set(q) if q is not None else set([])
    #     if len(rel) > 0:
    #         r = rel2quad.get(rel[0])
    #         if r is not None:
    #             subset_e = subset_e.intersection(r)
        
    #     subset = set([])
    #     if len(_time) > 0:
    #         for t in _time:
    #             for quad in subset_e:
    #                 if str(t) in str(quad[-1]):
    #                     subset.add(tuple(quad))
    #     else:
    #         subset = subset_e
    #     print(f'two_subset_e: {len(subset)}')
    #     return subset

    def extract_quad_from_anchor_no_answer(self, question, time, rel, anchor_event):
        question = question.lower()
        entities = []
        if len(anchor_event)>0:
            entities.append(anchor_event[0])
            entities.append(anchor_event[2])
        _time = []
        if 'month' in question:
            time_level = 'month'
            head2quad = self.head2quad_tm
            tail2quad = self.tail2quad_tm
            rel2quad = self.rel2quad_tm
            _time = []
            for t in time:
                _time.append(t[:7])
        elif 'year' in question:
            time_level = 'year'
            head2quad = self.head2quad_ty
            tail2quad = self.tail2quad_ty
            rel2quad = self.rel2quad_ty
            # print(len(time))
            for t in time:
                # print(len(time))
                _time.append(t[:4])
        else:
            time_level = 'day'
            head2quad = self.head2quad
            tail2quad = self.tail2quad
            rel2quad = self.rel2quad
            _time = time
        # print(f'time_level: {time_level}')
        subset_e = set([])
        for e in entities:
            hq = head2quad.get(e)
            tq = tail2quad.get(e)
            q = (set(hq) if hq is not None else set([])) | (set(tq) if tq is not None else set([]))
            subset_e = subset_e | set(q) if q is not None else set([])
        if len(rel) > 0:
            r = rel2quad.get(rel[0])
            if r is not None:
                subset_e = subset_e.intersection(r)
        
        subset = set([])
        if len(_time) > 0:
            for t in _time:
                for quad in subset_e:
                    if str(t) in str(quad[-1]):
                        subset.add(tuple(quad))
        else:
            subset = subset_e

        filter_quads = []
        if len(anchor_event) == 0:
            return subset
        target_time = datetime.strptime(anchor_event[-1], '%Y-%m-%d')

        for quad in subset:
            if time_level == 'year':
                event_time = datetime.strptime(quad[-1], '%Y')
            elif time_level == 'month':
                event_time = datetime.strptime(quad[-1], '%Y-%m')
            else:
                event_time = datetime.strptime(quad[-1], '%Y-%m-%d')
            if 'before' in question:
                if event_time < target_time:
                    filter_quads.append(quad)
            if 'after' in question:
                if event_time > target_time:
                    filter_quads.append(quad)
       
        if time_level == 'year':
            anchor_event[-1] = anchor_event[-1][:4]
        if time_level == 'month':
            anchor_event[-1] = anchor_event[-1][:7]
        filter_quads.append(tuple(anchor_event))
        quads = self.sort_by_date(filter_quads, time_level,ascending=True)
        if 'before' in question:
            quads = self.sort_by_date(filter_quads, time_level,ascending=False)
            quads = quads[:15]
        if 'after' in question:
            quads = quads[:15]
        return quads
        
    def extract_quad_from_anchor(self, nodes, question, entities, time, rel, anchor_event):
        question = question.lower()
        ent_tmp = []
        if len(anchor_event)>0:
            ent_tmp.append(anchor_event[0])
            ent_tmp.append(anchor_event[2])
            entities = ent_tmp
        print(f'entities: {entities}, time: {time}, rel: {rel}')
        _time = []
        if 'month' in question:
            time_level = 'month'
            head2quad = self.head2quad_tm
            tail2quad = self.tail2quad_tm
            rel2quad = self.rel2quad_tm
            _time = []
            for t in time:
                _time.append(t[:7])
        elif 'year' in question:
            time_level = 'year'
            head2quad = self.head2quad_ty
            tail2quad = self.tail2quad_ty
            rel2quad = self.rel2quad_ty
            print(len(time))
            for t in time:
                print(len(time))
                _time.append(t[:4])
        else:
            time_level = 'day'
            head2quad = self.head2quad
            tail2quad = self.tail2quad
            rel2quad = self.rel2quad
            _time = time
        print(f'time_level: {time_level}')
        subset_e = set([])
        for e in entities:
            hq = head2quad.get(e)
            tq = tail2quad.get(e)
            q = (set(hq) if hq is not None else set([])) | (set(tq) if tq is not None else set([]))
            subset_e = subset_e | set(q) if q is not None else set([])
        if len(rel) > 0:
            r = rel2quad.get(rel[0])
            if r is not None:
                subset_e = subset_e.intersection(r)

        subset = set([])
        if len(_time) > 0:
            for t in _time:
                for quad in subset_e:
                    if str(t) in str(quad[-1]):
                        subset.add(tuple(quad))
        else:
            subset = subset_e

        filter_quads = []
        if len(anchor_event) == 0:
            return subset, []
        target_time = datetime.strptime(anchor_event[-1], '%Y-%m-%d')

        for quad in subset:
            if time_level == 'year':
                event_time = datetime.strptime(quad[-1], '%Y')
            elif time_level == 'month':
                event_time = datetime.strptime(quad[-1], '%Y-%m')
            else:
                event_time = datetime.strptime(quad[-1], '%Y-%m-%d')
            if 'before' in question:
                if event_time < target_time:
                    filter_quads.append(quad)
            if 'after' in question:
                if event_time > target_time:
                    filter_quads.append(quad)

        if time_level == 'year':
            anchor_event[-1] = anchor_event[-1][:4]
        if time_level == 'month':
            anchor_event[-1] = anchor_event[-1][:7]
              
        filter_quads.append(tuple(anchor_event))
        quads = self.sort_by_date(filter_quads, time_level,ascending=True)
            
        if 'before' in question:
            quads = self.sort_by_date(filter_quads, time_level,ascending=False)
            quads = quads[:15]
        if 'after' in question:
            quads = quads[:15]
        q_for_nodes = defaultdict(set)
        for node in nodes:
            h = head2quad.get(node)
            t = tail2quad.get(node)
            ts = self.time2quad.get(node)
            
            relactive_quad = (set(h) if h is not None else set([])) | (set(t) if t is not None else set([])) | (set(ts) if ts is not None else set([]))
            if relactive_quad is not None and quads is not None:
                question_rel = relactive_quad.intersection(quads)
                q_for_nodes[node] = question_rel
        return quads, q_for_nodes
    def extract_quad_from_anchor_old(self, nodes, question, entities, time, rel, anchor_event):
        question = question.lower()
        print(f'entities: {entities}, time: {time}, rel: {rel}')
        _time = []
        if 'month' in question:
            time_level = 'month'
            head2quad = self.head2quad_tm
            tail2quad = self.tail2quad_tm
            rel2quad = self.rel2quad_tm
            _time = []
            for t in time:
                _time.append(t[:7])
        elif 'year' in question:
            time_level = 'year'
            head2quad = self.head2quad_ty
            tail2quad = self.tail2quad_ty
            rel2quad = self.rel2quad_ty
            print(len(time))
            for t in time:
                print(len(time))
                _time.append(t[:4])
        else:
            time_level = 'day'
            head2quad = self.head2quad
            tail2quad = self.tail2quad
            rel2quad = self.rel2quad
            _time = time
        print(f'time_level: {time_level}')
        subset_e = set([])
        for e in entities:
            hq = head2quad.get(e)
            tq = tail2quad.get(e)
            q = (set(hq) if hq is not None else set([])) | (set(tq) if tq is not None else set([]))
            subset_e = subset_e | set(q) if q is not None else set([])
        if len(rel) > 0:
            r = rel2quad.get(rel[0])
            if r is not None:
                subset_e = subset_e.intersection(r)
        print(f'subset_e: {len(subset_e)}')
        subset = set([])
        if len(_time) > 0:
            for t in _time:
                for quad in subset_e:
                    if str(t) in str(quad[-1]):
                        subset.add(tuple(quad))
        else:
            subset = subset_e
        # 过滤锚点事件时间关系
        filter_quads = []
        target_time = datetime.strptime(anchor_event[-1], '%Y-%m-%d')
        print(f'target_time: {target_time}')
        for quad in subset:
            if time_level == 'year':
                event_time = datetime.strptime(quad[-1], '%Y')
            elif time_level == 'month':
                event_time = datetime.strptime(quad[-1], '%Y-%m')
            else:
                event_time = datetime.strptime(quad[-1], '%Y-%m-%d')
            if 'before' in question:
                if event_time < target_time:
                    filter_quads.append(quad)
            if 'after' in question:
                if event_time > target_time:
                    filter_quads.append(quad)
        # print(f'filteres_quads: {len(filter_quads)}')
        # print("--------------")
        if time_level == 'year':
            anchor_event[-1] = anchor_event[-1][:4]
        if time_level == 'month':
            anchor_event[-1] = anchor_event[-1][:7]
        filter_quads.append(tuple(anchor_event))
        quads = self.sort_by_date(filter_quads, time_level,ascending=True)
        if 'before' in question:
            quads = self.sort_by_date(filter_quads, time_level,ascending=False)
            quads = quads[:15]
        if 'after' in question:
            quads = quads[:15]
        q_for_nodes = defaultdict(set)
        for node in nodes:
            h = head2quad.get(node)
            t = tail2quad.get(node)
            ts = self.time2quad.get(node)
            
            relactive_quad = (set(h) if h is not None else set([])) | (set(t) if t is not None else set([])) | (set(ts) if ts is not None else set([]))
            if relactive_quad is not None and quads is not None:
                question_rel = relactive_quad.intersection(quads)
                q_for_nodes[node] = question_rel
        return quads, q_for_nodes
        

        
    def pre_extract_quads(self, question_entities,time, rel, time_level):
        if time_level == "month":
            head2quad = self.head2quad_tm
            tail2quad = self.tail2quad_tm
            rel2quad = self.rel2quad_tm
        elif time_level == "year":
            head2quad = self.head2quad_ty
            tail2quad = self.tail2quad_ty
            rel2quad = self.rel2quad_ty
        else:
            head2quad = self.head2quad
            tail2quad = self.tail2quad
            rel2quad = self.rel2quad
        print(f'question_entities: {question_entities}, time: {time}')
        print(f'rel : {rel}')
        subset_e = set([])
        for i,e in enumerate(question_entities):

            hq = head2quad.get(e)
            tq = tail2quad.get(e)
            q = (set(hq) if hq is not None else set([])) | (set(tq) if tq is not None else set([]))
            subset_e = subset_e | set(q) if q is not None else set([])
            if len(rel) > 0 and len(time) == 0:
                r = rel2quad.get(rel[0])
                if r is not None:
                    subset_e = subset_e.intersection(r)
            
        subset = set([])
        if len(time) > 0:
            for t in time:
                for quad in subset_e:
                    if str(t) in str(quad[-1]):
                        subset.add(tuple(quad))
        else:
            subset = subset_e
        print(f'subset_len: {len(subset)}')
        return subset

class KG_Process_TQ(KG_Process):
    def read_txt_quadruples_space(self, file_path):
        quadruples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    elements = line.split('\t')
                    if len(elements) == 5:
                        quadruples.append(tuple(elements))
        except FileNotFoundError:
            print(f"File {file_path} doesn't exist.")
        return quadruples
   
    def sort_by_date(self, data, ascending=True):
        return sorted(data, key=lambda x: int(x[-2]), reverse=not ascending)
        
    def build_kg(self):
        print("Begin to build knowledge graph")
        for head, rel, tail, ts1, ts2 in tqdm(self.kg_quads):
           
            quad = []
            quad.append(head)
            quad.append(rel)
            quad.append(tail)
            quad.append(ts1)
            quad.append(ts2)
            
            
            self.head2quad[head].add(tuple(quad))
            self.tail2quad[tail].add(tuple(quad))
            self.rel2quad[rel].add(tuple(quad))
            self.time2quad[ts1].add(tuple(quad))
            self.time2quad[ts2].add(tuple(quad))

    def find_two_jump_quads(self, question_entities, time):
        one_jump_elements = []
        subset_e = set([])
        for e in question_entities:
            if self.head2quad.get(e) is not None:
                subset_e = subset_e | self.head2quad.get(e)
        subset = set([])
        if len(time) > 0:
            for t in time:
                for quad in subset_e:
                   
                    t_from = int(quad[-2])
                    t_to = int(quad[-1])
                    if len(t) >= t_from and int(t) <= t_to:
                        subset.add(tuple(quad))
        else:
            subset = subset_e
        two_jump_subset = subset
        for s in subset:
            for e in s[:-2]:
                one_jump_elements.append(e)
        for e in one_jump_elements:
            if self.head2quad.get(e) is not None:
                two_jump_subset = two_jump_subset | self.head2quad.get(e)
            if self.tail2quad.get(e) is not None:
                two_jump_subset = two_jump_subset | self.tail2quad.get(e)
            if self.rel2quad.get(e) is not None:
                two_jump_subset = two_jump_subset | self.rel2quad.get(e)
        result = set([])
        if len(time) > 0:
            for t in time:
                for quad in two_jump_subset:
                    t_from = int(quad[-2])
                    t_to = int(quad[-1])
                    if t >= t_from and t <= t_to:
                        result.add(tuple(quad))
        if len(result)==0:
            result = two_jump_subset
        print(f'two_jump_subset length: {len(result)}')
        return result

    def pre_extract_quads(self, question_entities,time):
        print(f'question_entities: {question_entities}, time: {time}')
        subset_e = set([])
        for e in question_entities:
            if self.head2quad.get(e) is not None:
                subset_e = subset_e | self.head2quad.get(e)
            if self.tail2quad.get(e) is not None:
                subset_e = subset_e | self.tail2quad.get(e)
            
                
        subset = set([])
        if len(time) > 0:
            for t in time:
                for quad in subset_e:
                    # if quad[-2] == 'unknown' and quad[-1] == 'unknown':
                    #     subset.add(tuple(quad))
                    # elif quad[-2] != 'unknown'and quad[-1] == 'unknown':
                    #     if int(t) >= int(quad[-2]):
                    #         subset.add(tuple(quad))
                    # elif quad[-2] == 'unknown'and quad[-1] != 'unknown':
                    #     if int(t) <= int(quad[-1]):
                    #         subset.add(tuple(quad))
                    # else:
                    t_from = int(quad[-2])
                    t_to = int(quad[-1])
                    if int(t) >= t_from and int(t) <= t_to:
                        subset.add(tuple(quad))
        else:
            subset = subset_e
        print(f'subset_len: {len(subset)}')
        return subset
    def extract_kg_quads_by_anchor_nodes(self, nodes):
        subset_e = set([])
        for n in nodes:
            if self.head2quad.get(n) is not None:
                subset_e = subset_e | self.head2quad.get(n)
            if self.tail2quad.get(n) is not None:
                subset_e = subset_e | self.tail2quad.get(n)
            # if self.rel2quad.get(n) is not None:
            #     subset_e = subset_e | self.rel2quad.get(n)
        facts = self.sort_by_date(subset_e, ascending=True)
        return facts

    def extract_kg_quads_by_anchor(self, question,question_entities,time, anchor_event_list):
        rel = ''
        
        if len(anchor_event_list)>0:
            question_entities = set([])
            anchor_event = anchor_event_list
            print(f'anchor_event: {anchor_event}')
            question_entities.add(anchor_event[0])
            rel = anchor_event[1]
            if len(anchor_event) == 5:
                question_entities.add(anchor_event[2])
        print(f'question_entities: {question_entities}, time: {time}')
        subset_e = set([])
        
        for e in question_entities:
            if self.head2quad.get(e) is not None:
                subset_e = subset_e | self.head2quad.get(e)
            if self.tail2quad.get(e) is not None:
                subset_e = subset_e | self.tail2quad.get(e)
        if len(rel) > 0:
            if self.rel2quad.get(e) is not None:
                subset_e = subset_e | self.rel2quad.get(e)
                
        subset = set([])
        if len(time) > 0:
            for t in time:
                for quad in subset_e:
                    t_from = int(quad[-2])
                    t_to = int(quad[-1])
                    if int(t) >= t_from and int(t) <= t_to:
                        subset.add(tuple(quad))
        else:
            subset = subset_e
        print(f'subset_len: {len(subset)}')
        facts = self.sort_by_date(subset, ascending=True)
        if 'before' in question:
            facts = self.sort_by_date(subset, ascending=False)
            facts = facts[:15]
        if 'after' in question:
            facts = facts[:15]
        
        return self.filter_unknown_time(facts)
        
        # if len(anchor_event_list)==0:
            
        # # 过滤锚点事件时间关系
        # anchor_event = anchor_event_list[0]
        # filter_quads = []
        # from_time_known = False
        # to_time_unknown = False
        # target_from_time = '1'
        # target_to_time = '2096'
        # if anchor_event[-2]!='1':
        #     target_from_time = datetime.strptime(anchor_event[-2], '%Y')
        #     from_time_unknown = True
        # if anchor_event[-1]!='2096':
        #     target_to_time = datetime.strptime(anchor_event[-1], '%Y')
        #     to_time_unknown = True
        
        # for fact in subset:
        #     if 'before' in question and from_time_known is True:
        #         if event_time < target_from_time:
        #             filter_quads.append(quad)
        #     if 'after' in question and to_time_unknown is True:
        #         if event_time > target_to_time:
        #             filter_quads.append(quad)
        # filter_quads.append(tuple(anchor_event))
        # for e in filter_quads:
        #     print(e)
        # facts = self.sort_by_date(filter_quads, ascending=True)
        # if 'before' in question:
        #     facts = self.sort_by_date(filter_quads, ascending=False)
        #     facts = quads[:15]
        # if 'after' in question:
        #     facts = quads[:15]
        
        # return self.filter_unknown_time(facts)
    def filter_unknown_time(self, fact_set):
        subset_filtered = []
        for q in fact_set:
            if len(q)==5:
                head,rel,tail,t1,t2 = q
                if t1=='1':
                    t1='unknown'
                if t2=='2916':
                    t2='unknown'
                subset_filtered.append(tuple([head,rel,tail,t1,t2]))
            if len(q)==4:
                head,rel,t1,t2 = q
                if t1=='1':
                    t1='unknown'
                if t2=='2916':
                    t2='unknown'
                subset_filtered.append(tuple([head,rel,t1,t2]))
        return subset_filtered
        

    def extract_kg_quads(self, nodes, question_entities,time):
        if len(nodes)==0:
            print("No candidate answers!")
            return []
        print(f'question_entities: {question_entities}, time: {time}')
        subset_e = set([])
        for e in question_entities:
            if self.head2quad.get(e) is not None:
                subset_e = subset_e | self.head2quad.get(e)
            if self.tail2quad.get(e) is not None:
                subset_e = subset_e | self.tail2quad.get(e)
            
                
        subset = set([])
        if len(time) > 0:
            for t in time:
                for quad in subset_e:
                    if quad[-2] == 'unknown' and quad[-1] == 'unknown':
                        subset.add(tuple(quad))
                    elif quad[-2] != 'unknown'and quad[-1] == 'unknown':
                        if t >= int(quad[-2]):
                            subset.add(tuple(quad))
                    elif quad[-2] == 'unknown'and quad[-1] != 'unknown':
                        if t <= int(quad[-1]):
                            subset.add(tuple(quad))
                    else:
                        t_from = int(quad[-2])
                        t_to = int(quad[-1])
                        if t >= t_from and t <= t_to:
                            subset.add(tuple(quad))
        else:
            subset = subset_e
        print(f'subset_len: {len(subset)}')
        # filter_q = defaultdict(set)
        # for node in nodes:
        #     if isinstance(node, int):
        #         for e in subset:
        #             if node <= int(e[-1]) and node >= int(e[-2]):
        #                 filter_q[node].add(e)
        #     else:
        #         h = self.head2quad.get(node)
        #         t = self.tail2quad.get(node)
        #         relactive_quad = (set(h) if h is not None else set([])) | (set(t) if t is not None else set([]))
        #         if relactive_quad is not None and subset is not None:
        #             question_rel = relactive_quad.intersection(subset)
        #             filter_q[node] = question_rel
            
            # if len(filter_q[node])>0:
            #     print("------")
            #     print(node)
            #     print(len(filter_q[node]))
        return subset
            