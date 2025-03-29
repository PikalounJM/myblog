---
title: "VOC Topic Modeling"
description: Voice of Customers, Keyword extraction, LDA
category: projects
dateString: June 2024 - July 2024
cover:
  image: "img/topic.PNG"
  alt:
  caption:
  relative: 
showtoc: false
draft: false
weight: 202
---

### Coding
The complete code can be found in 🔗 [Github](https://github.com/PikalounJM/Text-Mining/blob/main/VOC_Topic%20Modeling.ipynb)

### Backgroud
- To identify on/offline customer inconveniences and improve usability through VOC. 

### Dataset & Tool
- 139,010 **using Python**

### Method
- Complaint Keyword grouping using Topic modeling

&nbsp;

### ✏️ Code Example

**1) 텍스트 전처리(Text preprocessing)**
- 데이터 치환 및 정규표현식 적용

```python
neg_19 = pd.DataFrame(before[before['성격']=='불만'][['사업장','내용']])
neg_19.reset_index(drop=True, inplace=True) #index 초기화

#불용어 불러오기
stopword = pd.read_excel('C:/Users/user/Desktop/stopword_list.xlsx')
#stopword.rename(columns={0:'불용어'}, inplace=True)

replace_list = pd.read_excel('replace_list.xlsx')
replace_list.loc[377] = ['콘도','리조트']
replace_list.loc[378] = ['방','객실']
replace_list.loc[379] = ['소리','소음']
replace_list.loc[380] = ['부페','뷔페']
replace_list.loc[381] = ['프런트','프론트']
replace_list.loc[382] = ['숙박','투숙']
replace_list.loc[383] = ['숙소','객실']
replace_list.loc[384] = ['입실','투숙']

#데이터 치환
def replace_word(내용):
    for i in range(len(replace_list['before_replacement'])):
        try:
            #치환할 단어가 있는 경우에만 데이터 치환 수행
            if replace_list['before_replacement'][i] in 내용:
                내용 = 내용.replace(replace_list['before_replacement'][i], replace_list['after_replacement'][i])
        except Exception as e:
            print(f"Error 발생 / 에러명: {e}")
    return 내용

from tqdm import tqdm
neg_19['review_prep'] = ''
review_replaced_list = []
for 내용 in tqdm(neg_19['내용']):
    review_replaced = replace_word(str(내용))
    review_replaced_list.append(review_replaced)

neg_19['review_prep'] = review_replaced_list

#list로 변환(df상태에서는 전처리가 어려움)
neg_19_ = neg_19.review_prep.values.tolist()

#정규표현식 적용
import re

regex = []

for i in range(len(neg_19)):
  text = re.sub('[^0-9ㄱ-힣]',' ', str(neg_19_[i]))
  text = re.sub(' +',' ', text)
  regex.append(text)
```
&nbsp;
- 추출 단어 중 최소/최대 토큰 개수 정하기
- 유의미한 분석을 위해 각 리뷰에서 추출된 명사의 개수가 3개 ~ 15개 이하인 리뷰만 추출

```python
### 추출 명사 중 최소/최대 토큰 개수 정하기
min_token = 3
max_token = 15

df_19 = df_19[df_19['noun'].apply(lambda tokens: min_token <= len(tokens) <= max_token)][['noun','review','사업장']]
df_21 = df_21[df_21['noun'].apply(lambda tokens: min_token <= len(tokens) <= max_token)][['noun','review','사업장']]
df_22 = df_22[df_22['noun'].apply(lambda tokens: min_token <= len(tokens) <= max_token)][['noun','review','사업장']]
df_23 = df_23[df_23['noun'].apply(lambda tokens: min_token <= len(tokens) <= max_token)][['noun','review','사업장']]
```
&nbsp;

**2) LDA 사전 제작**

- 최적의 토픽 수 찾기(Perplexity & Coherence score)
- **Perplexity**: 주제의 복잡성으로 score가 높을 수록 각 토픽들이 문서를 잘 반영하지 못함
- **Coherence**: 주제의 일관성을 score가 높을 수록 해당 토픽 간의 주제들이 서로 일관성이 있음

```python
PASSES = 10

perplexity_values = []

for i in range(2,8):
  ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=id2word, passes = PASSES)
  perplexity_values.append(ldamodel.log_perplexity(corpus))

x = range(2,8)
plt.plot(x, perplexity_values)
plt.xlabel('Number of topics')
plt.ylabel('Perplexity score')
plt.show()

from gensim.models import CoherenceModel

PASSES = 10
coherence_values = []

for i in range(2,8):
  ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=id2word, passes = PASSES, random_state=10)
  coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=id2word)
  coherence_lda = coherence_model_lda.get_coherence()
  coherence_values.append(coherence_lda)

x = range(2,8)
plt.plot(x, coherence_values)
plt.xlabel('Number of topics')
plt.ylabel('coherence score')
plt.show()
```
&nbsp;
![Graph](/img/graph.PNG)


**3) 문서별 토픽 번호 분류**
```python
#문서별 토픽화
result_23 = []

for i, doc in enumerate(corpus_23):
    topic_probs = ldamodel.get_document_topics(doc)
    topic_probs = sorted(topic_probs, key= lambda x: x[1], reverse=True)
    top_topic = topic_probs[0]
    result_23.append({
    '문서 번호': i,
    'Topic': top_topic[0],
    '주제 확률': top_topic[1]
    })

topic_23 = pd.DataFrame(result_23)
topic_22['Topic'] = topic_23['Topic'].replace({0:1, 1:2, 2:3})
topic_23
```
&nbsp;
### Insight Report
- Developed a storyline, presentation decks and briefing documents that support insights.
![Report Ime](/img/voc.PNG)
