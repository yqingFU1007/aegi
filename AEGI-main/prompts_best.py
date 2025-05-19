from openai import OpenAI
import re

PROMPT_TQ_ANSWER = """
You are a ISO Temporal Expert! Generate answers STRICTLY following temporal logic rules. FINAL LINE MUST BE: [brief_word/phrase]

Input Parameters:
Question: {question}
Relevant Facts: {facts}

Core Temporal Rules:
[Critical Constraints]
1.Ignore invalid time ranges (1-2916)
2.Event order determined by start_time
3.Chain analysis for multi-event queries
4.Count distinct entities for numerical questions
5.Time format: Use ONLY year(s) or comma-separated ranges
6.Output 'None' when can't get answer from facts
7.Output [year],[year] if output duration,for example if the answer is "1905-1919", output "1905,1919"

Required Output Format:
The FINAL line must be(If you have several answers, split by ','): 
[your_answer]

Examples:
Example Input:
Question: When was the Beijing Olympics held?
Facts: (Beijing, host, Summer_Olympics, 2008, 2009)
Example Output:
[2008]
"""

PROMPT_TQ_RERANK = """
You are a Strict Format Ranking Engine. Rerank answers using temporal logic and facts. ONLY output 2 lines: 1) Rerank result in ["a","b"] format, 2) Confidence scores in {x,y} format. NO explanations!


Question: {question}

Relevant Facts: {question_relevant_facts}

Candidate Answers: {answers}

Confidence Score evaluation standards is:
1.Fact Consistency (1-5): 5=fully supported, 1=conflict
2.Relevance (1-5): 5=direct answer, 1=irrelevant

Score each evaluation dimension 1-5, and the total score is the sum of the two part of scores(full score is 10).

Reasoning rules:
1.Ignore invalid time ranges (1-2916)
2.Event order determined by start_time
3.Chain analysis for multi-event queries
4.Count distinct entities for numerical questions
5.Time format: Use ONLY year(s) or comma-separated ranges
6.If NO facts relate to an answer → score=0

REQUIRED OUTPUT FORMAT:
Rerank result:["answer1","answer2","answer3,"answer4","answer5","answer6","answer7","answer8","answer9","answer10"](You should rerank and output all the candidate answers)
Confidence scores:{score1,score2,score3,score4,score5,score6,score7,score8,score9,score10}(You should output all the scores of the candidate answers after rerank)

DO NOT OUTPUT ANY OTHER TEXT! [/INST]
"""


PROMPT_MULTITQ="""
You are a Knowledge-Enhanced Ranking Specialist,tasked with precisely reordering answers using knowledge graph facts.

The Question is: 
{question}

There are some facts relevant to the question.
{question_relevant_facts}


Candidate_Answers:{answers}


For each answer, we have facts from knowledge graph(each fact is a quadruple(head_entity, relation, tail_entity, time), means on "time", happens (head_entity, relation, tail_entity)):
{facts}


Referring to the facts and origin order, you should rerank the Candidate_Answers according to the new answer confidence level to get the right answer of each question.

You should also give a list of confidence score of each question, evaluation standards is:
1.Fact Consistency: Compare with the provided fact quintuples. Does it conflict?
2.Relevance: Directly addresses the question? (Avoid irrelevant answers)
Score each evaluation dimension 1-5, and the total score is the sum of the two part of scores.

Time format:"yyyy-mm-dd" or "yyyy-mm" or "yyyy"


If you are sure there is no correct answer in the Candidate_Answers, give low score to all of the Candidate_Answers.


You should reorder all the Candidate_Answers and output the Candidate_Answers list after reranking(if you keep the original order, the rerank list is the same as the Candidate_Answers), all the answers in the output list should in the Candidate_Answers list.And you should output a list of conficence score of each candidate answer after reranking,you should not give any other information in the output.
If you don't have enough fact supporting you to rerank, output the original order.
ONLY output 2 lines: 1) Rerank result in ["a","b"] format, 2) Confidence scores in {x,y} format. NO explanations!
REQUIRED OUTPUT FORMAT:
Rerank result:["answer1","answer2","answer3,"answer4","answer5","answer6","answer7","answer8","answer9","answer10"](You should rerank and output all the candidate answers)
Confidence scores:{score1,score2,score3,score4,score5,score6,score7,score8,score9,score10}(You should output all the scores of the candidate answers after rerank)
"""

PROMPT_MULTI_ANSWER = """
You are a ISO Temporal Expert! Generate answers STRICTLY following temporal logic rules. FINAL LINE MUST BE: [brief_word/phrase]


Input Parameters:
Question: {question}
Relevant Facts: {facts}

Reasoning Template:
1.Extracted relevant Temporal Facts(Events),you should extract all the quadruples relevant to the question(all the quadruples whose entity or relation is similar as the entities or relations in the question and all the quadruples whose time is relevant to the question) from the given facts:
(Entity 1, relation1, entity 2, time1)
(Entity 3, relation2, entity 4, time2)
...

2.Timeline Construction

3.Temporal Logic Applied:
Rule 1: Transitive before/after/first/last
Rule 2: If time1 < time2,then event1(Entity 1, relation1, entity 2) happens before event2(Entity 3, relation2, entity 4, time2)
Rule 3: If time1 > time2,then event1(Entity 1, relation1, entity 2) happens after event2(Entity 3, relation2, entity 4, time2)
Rule 4: If there are several events(events3(Entity 5, relation3, entity 6, time3),event4(Entity 7, relation4, entity 8, time4),...) happen after event1(Entity 1, relation1, entity 2, time1), it means time3 > time1, time4>time1, if time3 is the smallest among these envents happen after event1, than event3 is the first happen after event1.
Rule 5: If there are several events(events3(Entity 5, relation3, entity 6, time3),event4(Entity 7, relation4, entity 8, time4),...) happen before event1(Entity 1, relation1, entity 2, time1), it means time3 < time1, time4<time1, if time3 is the largest among these envents happen before event1, than event3 is the last happen after event1.

If there are more than one correct answers, give all correct answers.
If the given fact can't reasoning the correct answer, output "None"
If your answer is a time, please output in ISO format, for example "2014-02-02", "2014-02", "2014", don't give "February"


Required Output Format:
The FINAL line must be(If you have several answers, split by ','): 
[your_answer]

Examples:
Example Input:
Question: When was the Beijing Olympics held?
Facts: (Beijing, host, Summer_Olympics, 2008)
Example Output:
[2008]

Example Input:
Question: After the Cabinet Council of Ministers of Kazakhstan, who was the first to express the intention to negotiate with China?
Facts: ('Cabinet / Council of Ministers / Advisors (Kazakhstan)', 'Express intent to meet or negotiate', 'China', '2010-02-20')
('Envoy (United States)', 'Express intent to meet or negotiate', 'China', '2010-02-22')
('Citizen (North Korea)', 'Express intent to meet or negotiate', 'China', '2010-02-22')
('South Korea', 'Express intent to meet or negotiate', 'China', '2010-02-22')
('Rupiah Banda', 'Express intent to meet or negotiate', 'China', '2010-02-23')
After the Cabinet Council of Ministers of Kazakhstan, Envoy (United States),South Korea and Citizen (North Korea) all express the intention to negotiate with China in the same day on 2010-02-22, so there are three answers.
Example Output:
[Envoy (United States),Citizen (North Korea),South Korea]

Example Input:
Question: Which country condemned China in August 2013?
Facts :('China', 'Accuse', 'Envoy (United States)', '2013-08-29')
The fact indicate "China accuse Envoy (United States) on 2013-08-29", but the question is Which country condemned China, not China condemned which country, so there is no fact can support the reasoning, the answer is 'None'.
Example Output:
[None]
"""

client = OpenAI(
    base_url='BASE_URL_GPT',
    api_key='sk-xxxxxxxxx'
)

deepseek_client = OpenAI(
    base_url='BASE_URL_DEEPSEEK',
    api_key='sk-xxxxxxxxx'
)

deepseek_client_multi = OpenAI(
    base_url='BASE_URL_DEEPSEEK',
    api_key='sk-xxxxxxxxx'
)

def deepseek_v3(prompt):
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Answer the question in a brief word or phrase. Don't use a complete sentence!If your answer is a time expression, only give the year like \"1989\". "},
            {"role": "user", "content": prompt}
          ],
        stream=False
    )
    return response.choices[0].message.content

def deepseek_v3_multi(prompt):
    response = deepseek_client_multi.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Answer the question a brief word or phrase. Don't use a complete sentence!If your answer is a time expression, give the date in the form as \"1989-01-01\".Give me the top 10 answers split by ','."},
            {"role": "user", "content": prompt}
          ],
        stream=False
    )
    return response.choices[0].message.content


def gpt_4o_mini(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "user", "content": prompt}
      ],
        temperature=0.0,
        seed=319
    )
    return response.choices[0].message.content

def extract_answer(text):
    if text is None:
        return "None"
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return "None"
        

def prompt_tq_answer(question, facts):
    prompt = PROMPT_TQ_ANSWER
    prompt = prompt.replace('{question}',str(question))
    prompt = prompt.replace('{facts}',str(facts))
    return extract_answer(gpt_4o_mini(prompt))

def prompt_tq_rerank(question, q_facts, condidates):
    prompt = PROMPT_TQ_RERANK
    prompt = prompt.replace('{answers}',str(condidates))
    prompt = prompt.replace('{question}',str(question))
    prompt = prompt.replace('{question_relevant_facts}',str(q_facts))
    return gpt_4o_mini(prompt)


def prompt_multi_rerank(condidates, question,facts, q_facts,answer_type):
    prompt = PROMPT_MULTITQ
    prompt = prompt.replace('{answers}',str(condidates))
    prompt = prompt.replace('{question}',str(question))
    prompt = prompt.replace('{facts}',str(facts))
    prompt = prompt.replace('{question_relevant_facts}',str(q_facts))
    prompt = prompt.replace('{answer_type}',str(answer_type))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "user", "content": prompt}
      ],
        temperature=0.0,
        seed=319
    )
    return response.choices[0].message.content

def get_multi_prompt(question, facts):
    prompt = PROMPT_MULTI_ANSWER
    prompt = prompt.replace('{question}',str(question))
    prompt = prompt.replace('{facts}',str(facts))
    return prompt

    
def prompt_multi_answer(question, facts):
    prompt = PROMPT_MULTI_ANSWER
    prompt = prompt.replace('{question}',str(question))
    prompt = prompt.replace('{facts}',str(facts))
    return extract_answer(gpt_4o_mini(prompt))

    
