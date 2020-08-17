# Seq2seq  한영번역기

Sequence to Sequence Learning with Neural Networks 참조하여 논문 구현

1. 긴 문장에도 좋은 성능을 유지하기 위하여 LSTM을 사용하여  Decoder / Encoder를 분리하여 학습
2. Encoder는 마지막 hidden state에서 출력된 고정 크기의 벡터를  출력,문장 마지막에 <end>를 추가하여 출력 생성을 중단하도록 함.
3. BLEU score을 확용하여 기계번역의 성능을 측정함
4. 학습 과정에서 문장의 순서를 뒤집어서(reverse) 훈련하여 성능을 향상시켰다.



**논문의 특징**

- LSTM으로 Exploding/Vanishing gradients 현상을 방지

- 성능을 위해 4 layers

- 입력 문장의 단어 순서를 뒤집었다.

- 각 layer마다 GPU 머신으로 병렬 작업 수행함

  
  
  
  
  나는 lstm 도 못하고 시중의 gru로 겨우겨우 돌렸다.
  
  lstm은 차원문제가 자꾸 발생함.
  
  LSTM 은 2개의 INPUT
  
  GRU 는 3개의 INPUT
  
  BI-LSTM 은 4개 ??
  
  
  
  역시나 전처리가 매우 중요하고 GPU는 너무나 소중한 존재임을 .. ...
  
  Colab pro가 처음으로 필요했다 ..ㅠㅠ
  
  

## 전처리

###  -  Reverse 실행

```python
def preprocess_sentence(w):
  w = w.lower().strip()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^ㄱ-ㅣ가-힣a-zA-Z?.!,]+", " ", w)
  w = w.strip()

  # 예측의 시작과 끝을 표시하기 위해 <start> <end> 표시
  if len(re.findall('[ㄱ-ㅣ가-힣]+',w)) == 0:
    w = '<start> ' + w + ' <end>'
  else :
    # reverse
    w = '<start> ' + w[::-1] + ' <end>'
    # w = w[::-1]
  return w
```

### reverse 한 결과

```python
ko_sentence = '나는 매일 저녁 배트를 만나러 다락방으로 가요.'
en_sentence = "I go to the attic every evening to meet Bat."
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(ko_sentence))
```

```python
<start> i go to the attic every evening to meet bat . <end>
<start> . 요가 로으방락다 러나만 를트배 녁저 일매 는나 <end>
```



### - Reverse 실행 X

### preprocessing만 진행

```python
def preprocess_sentence(w):
  w = w.lower().strip()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  

  w = re.sub(r"[^ㄱ-ㅣ가-힣a-zA-Z?.!,]+", " ", w)
  w = w.strip()

  if len(re.findall('[ㄱ-ㅣ가-힣]+',w)) == 0:
    w = '<start> ' + w + ' <end>'
  return w
```

#### reverse 없는 결과

```python

def create_dataset(path, num_examples):
  df = pd.read_excel(path,sheet_name=1)
  col = df.columns
  lines = df[col[1]] +'\t'+ df[col[2]]
  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)
```

```python
en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])
```

<reverse결과>

```python
<start> i go to the attic every evening to meet bat . <end>
나는 매일 저녁 배트를 만나러 다락방으로 가요 .
```





### Tokenize 

 train / validation / test : 50000/13000/12000 크기

```PYTHON
num_examples = 75000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
input_tensor, target_tensor = input_tensor[:63000, 1:], target_tensor[:63000, 1:]
xtest_tensor = input_tensor[63000:,1:]
ytest_tensor =target_tensor[63000:,1:]

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
```

```PYTHON
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=13000,random_state=42)

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
```

```PYTHON
50000 50000 13000 13000
```



#### bleu score

```python
for i in range(63000,75001):
  ref = [],pred = []
  ref.append(df['en'].iloc[i]))
  pred.append(translate(df['ko'].iloc[i]))
```

```python
def sentences_to_bleu(ref, pred):
 """
 ref : 참고용 타겟 문장(학습용 영어 문장)
 pred : 예측 문장(번역 결과)
 """
 smoothie = SmoothingFunction().method4
 return bleu.sentence_bleu(ref, pred, smoothing_function=smoothie)
 sentences_to_bleu(ref,pred)
```

![image-20200817231208569](C:\Users\yu946\AppData\Roaming\Typora\typora-user-images\image-20200817231208569.png)

