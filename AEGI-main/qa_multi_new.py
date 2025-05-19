import torch
import torch.nn.functional as F
import torch.nn as nn
from qa_baselines import QA_MultiQA
from torch.nn.utils.rnn import pad_sequence

class QA_MultiQA_NEW(QA_MultiQA):
    def __init__(self, tkbc_model, args):
        super().__init__(tkbc_model, args)
        self.dynamic_weight = DynamicWeight(768)
        self.args = args

        self.merge = nn.Linear(768*2, 768)

        self.answer_type_classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        self.tau_predictor = nn.Linear(768, 1)

        self.answer_type_loss = nn.CrossEntropyLoss()
        self.factor = 0.5

        self.cross_attn = MultiHeadTimeAwareAttention(
            self.tkbc_embedding_dim, 
            num_heads=8,
            time_aware=True
        )
        # self.cross_attn = TypeAwareMultiHeadAttention(self.num_entities+self.num_times, self.tkbc_embedding_dim)
        # self.fusion_layer = FusionGate(self.tkbc_embedding_dim)

        self.time_aggregator = TimeAggregator(self.tkbc_embedding_dim, 768)

        # 实体转时间索引
        # self.ent2time = torch.load("/root/autodl-tmp/MultiTQ-main/data/MultiTQ/kg/ent2tsid.pt")
        # self.ent2time = self.ent2time.cuda()
        # print(self.ent2time)
        self.all_entity_emb = self.entity_time_embedding.weight[:self.num_entities].cuda()
        self.all_time_emb = self.entity_time_embedding.weight[self.num_entities:].cuda()
        # self.entity2time_emb = self.all_time_emb[self.ent2time].cuda()

    def forward_on_classify(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        answer_type_mask = self.answer_type_classifier(question_embedding)
        return answer_type_mask
    

    def pred_question_type(self, question_embedding):
        answer_type_mask = self.answer_type_classifier(question_embedding.detach())
        type_probs = F.gumbel_softmax(answer_type_mask, tau=2, hard=False)
        pred_labels = torch.argmax(type_probs, dim=1)
        return pred_labels

    def forward(self, a):
        # print(f'tails: {a[3]}')
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        heads = a[2].cuda()
        tails = a[3].cuda()
        
        
        # time_embedding = [self._get_multi_transformered_time_embedding(x.cuda()).unsqueeze(0) for x in a[4]]
        # time_embedding = torch.cat(time_embedding, dim=0)

        head_embedding = self.entity_time_embedding(heads)
        tail_embedding = self.entity_time_embedding(tails)

        if self.args.dataset_name == 'timequestions':
            times = a[4].cuda()
            time_embedding = self.entity_time_embedding(times)

        raw_time_ids = a[4]
        time_emb_list = []
        for sample_times in raw_time_ids:
            sample_time_emb = self.entity_time_embedding(torch.tensor(sample_times).cuda())  # [num_times, time_dim]
            time_emb_list.append(sample_time_emb)
            
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)

        # rel_tokenized = a[5].cuda()
        # rel_attention_mask = a[6].cuda()
        # rel_embdding = self.getQuestionEmbedding(rel_tokenized, rel_attention_mask)
        # q_r = torch.cat([question_embedding, rel_embdding], axis=1)
        # q_r = self.merge(q_r)
        # question_embedding = q_r

        if self.args.dataset_name == 'MultiTQ':
            padded_time_emb = pad_sequence(time_emb_list, batch_first=True)  # [batch, max_times, time_dim]
            agg_time_emb = self.time_aggregator(padded_time_emb, question_embedding)
            time_embedding = agg_time_emb
        
        
        type_logits = self.answer_type_classifier(question_embedding.detach())
        tau = 1.0 + torch.sigmoid(self.tau_predictor(question_embedding)) * 1.5
        type_probs = F.gumbel_softmax(type_logits, tau=2, hard=False)
        entity_mask = torch.sigmoid(2*type_probs[:, 0] - 1).unsqueeze(1)
        time_mask = torch.sigmoid(2*type_probs[:, 1] - 1).unsqueeze(1)

        relation_embedding = self.linear(question_embedding)
        
        
        # 注意力
        # question_aware_entity = self.cross_attn(relation_embedding, self.all_entity_emb, self.all_entity_emb, interpolate_time_embedding(self.all_time_emb, self.num_entities))

        # type_weights = torch.stack([entity_mask.squeeze(), time_mask.squeeze()], dim=1)
        # question_aware_entity = self.cross_attn(
        #     rel_merge, 
        #     self.entity_time_embedding.weight,
        #     self.entity_time_embedding.weight,
        #     type_weights.detach(),
        #     interpolate_time_embedding(self.all_time_emb, self.num_entities+self.num_times)
        # )

        
        # relation_embedding = torch.cat([relation_embedding, question_aware_entity], dim=1)
        # relation_embedding = self.fusion_layer(rel_merge, question_aware_entity)

        relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
        relation_embedding2 = self.dropout(self.bn2(self.linear2(relation_embedding)))
        
        scores_time = self.score_time(head_embedding, tail_embedding, relation_embedding1)

        
        scores_entity1 = self.score_entity(head_embedding, relation_embedding2, time_embedding)
        scores_entity2 = self.score_entity(tail_embedding, relation_embedding2, time_embedding)
        scores_entity = torch.maximum(scores_entity1, scores_entity2)

        # 动态分数权重
        # weights = self.dynamic_weight(question_embedding)
        # entity_weight = weights[:, 0].unsqueeze(1)
        # time_weight = weights[:, 1].unsqueeze(1)
        # scores_entity = scores_entity * entity_weight
        # scores_time = scores_time * time_weight
        
        # scores_entity = scores_entity * (entity_mask + self.factor * time_mask)
        # scores_time = scores_time * (time_mask + self.factor * entity_mask)
        scores_entity = scores_entity * entity_mask
        scores_time = scores_time * time_mask
        scores = torch.cat((scores_entity, scores_time), dim=1)
        return scores


class TimeAggregator(nn.Module):
    def __init__(self, time_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(time_dim, hidden_dim)
        
    def forward(self, time_emb, question_emb):
        # time_emb: [batch, num_times, time_dim]
        # question_emb: [batch, seq_len, hidden_dim]
        
        # 生成时间注意力权重
        query = self.query(question_emb)  # [batch, hidden_dim]
        keys = self.key(time_emb)                     # [batch, num_times, hidden_dim]
        
        # 计算注意力得分
        attn_scores = torch.bmm(keys, query.unsqueeze(-1))  # [batch, num_times, 1]
        attn_weights = F.softmax(attn_scores, dim=1)        # [batch, num_times, 1]
        
        # 加权聚合
        aggregated_time = torch.sum(time_emb * attn_weights, dim=1)  # [batch, time_dim]
        return aggregated_time

class MultiHeadTimeAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, time_aware=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.time_proj = nn.Linear(embed_dim, num_heads) if time_aware else None
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value, time_emb=None):
        # 时间感知注意力偏置
        if self.time_proj and time_emb is not None:
            time_bias = self.time_proj(time_emb).permute(1,0)
            time_bias = time_bias.unsqueeze(1)     # [num_heads, 1, key_len]
            time_bias = time_bias.expand(-1, query.size(0), -1)
            attn_output, _ = self.attention(
                query, key, value, 
                attn_mask=time_bias
            )
        else:
            attn_output, _ = self.attention(query, key, value)
            
        return self.layer_norm(query + attn_output)

class TypeAwareMultiHeadAttention(nn.Module):
    def __init__(self, all_emb_dim, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.time_proj = nn.Linear(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        # self.type_bias_proj = nn.Sequential(
        #     nn.Linear(2, 128),  # 输入entity/time权重
        #     nn.GELU(),
        #     nn.Linear(128, num_heads),
        #     nn.Tanh()
        # )
        self.type_bias_proj = nn.Linear(2, num_heads)
        
    def forward(self, query, key, value, type_weights, time_emb):
        batch_size = type_weights.size(0)
        question_bias = self.type_bias_proj(type_weights)  # [N, H]
        L = query.size(0) 
        S = key.size(0)   

        time_bias = self.time_proj(time_emb).permute(1,0)
        time_bias = time_bias.unsqueeze(1)     # [num_heads, 1, key_len]
        time_bias = time_bias.expand(-1, query.size(0), -1)
        question_bias = question_bias.view(batch_size, -1, 1)  # [N, H, 1, 1]
        question_bias = question_bias.expand(-1, -1, S).permute(1,0, 2)       # [N, H, L, S]

        attn_output, _ = self.attention(
            query, key, value, 
            attn_mask=question_bias  # 形状需为 [N*H, L, S]
        )
        return self.layer_norm(query + attn_output)

class FusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(dim*2, dim)
        
    def forward(self, a, b):
        combined = torch.cat([a, b], dim=-1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        return gate * transformed + (1 - gate) * a

def interpolate_time_embedding(time_embedding, target_size):

    time_3d = time_embedding.permute(1, 0).unsqueeze(0)

    interpolated = F.interpolate(
        time_3d,
        size=target_size,
        mode='linear',
        align_corners=False
    )

    interpolated = interpolated.squeeze(0).permute(1, 0)
    return interpolated

class DynamicWeight(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, question):
        return self.mlp(question)