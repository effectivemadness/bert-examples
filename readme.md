## NLP Examples using BERT

### BERT
- 양방향을 가지는 마스크 언어 모델.
- 빈칸 채우기 + 다음문장 예측으로 학습된 모델.
- 이미 해당 문제로 학습된 모델은 언어에 대한 전반적인 이해를 잘 하고 있고, 이렇게 사전 학습된 지식을 기반으로 다른 문제 해결에 사용될 수 있다.
- Pretrained model usage : 모델 중간의 특징을 뽑아 활용하는 방식(feature-based), 다른 문제를 해결하기 위한 최소한의 가중치를 추가해 모델을 추가 학습하는 방식(fine-tune)이 있음.
- BertTokenizer.from_pretrained(), tokenizer.encode_plus() ftn으로 인코딩.

- NER(개체명 인식 - Named Entity Recognition)
    - data : Naver NLP Challenge 2018's NER data
    - 감정 분석, 자연어 추론 등은 bert의 마지막 히든 벡터를 사용했지만, NER 에서는 문장의 모든 입력 값을 예측해야 하므로 모든 히든 벡터를 사용해야 함.
    - Bert Tokenizer를 통과하면서 단어가 쪼개지므로(ex. 이순신 → 이, ##순, ##신) 라벨들도 이렇게 쪼개지는것을 감안해서 개수를 늘려줘야 함.
    - bert 출력(768(embed) * 111(max_sent_len)) 에서 dense 레이어를 거쳐 (111(max_sent_len) * 30(classes))로 출력.
    - 정확하게 예측했는지 평가하기 위해 F1 score 구현
        - precision과 recall을 합쳐 평가.
        - $F1 score = 2\times \frac {Precision\times Recall}  {Precision + Recall}$
- Text Similarity
    - data : KorSTS(kakao)
    - quora 데이터와 다르게 classification이 아니라 Regression - 0~5 scale로 유사도 산정
    - 자연어 추론 모델과 유사하게 tokenizer시 text_pair에 두번째 문장 전달.
    - 마지막 dense layer는 1개의 출력(유사도 값)을 가짐.
    - 평가함수로 피어슨 상관계수 정의 - 두 변수의 공분산을 표준편차로 나눔
    - $r_{xy}=\frac {cov(X,Y)} {\sigma_X\sigma_Y}$
- 기계 독해
    - data : KorQuAD 1.0
    - 답변이 주로 지문에 나올 명사이므로 명사 토큰에 대한 분석.
    - 입력 : \<CLS> "질문" \<SEP> "지문" \<SEP>
    - 출력 : 답변 시작 인덱스, 답변 끝 인덱스
    - bert 출력값을 뽑아 답변의 시작을 찾는 레이어, 답변에 끝을 찾는 레이어 각각에 넣어줌.
    - flatten 후 softmax로 최대 값을 가지는 부분 찾기.

- "텐서플로 2와 머신러닝으로 시작하는 자연어 처리" 책의 [코드](https://github.com/NLP-kr/tensorflow-ml-nlp-tf2)를 기반으로 기록용으로 작성하였습니다.