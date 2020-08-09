# NLP

# Convolutional neural networks for sentence classification 논문 구현
## : CNN을 이용하여 네이버 영화리뷰를 논문의 MODEL로 자연어 처리 및 감상분석함.

- **CNN-rand** : baseline값으로 사용하기 위해 사용. 모든 단어 벡터를 임의의 값으로 초기화해서 사용했다.
- **CNN-static** : 앞서 말한 사전 학습된 word2vec 단어 벡터를 사용한 모델이다.
- **CNN-non-static** : 위의 모델과 같이 학습된 벡터를 사용했지만 각 task에서 벡터값은 update된다.
- **CNN-multichannel** : architecture 소개 부분에서 나왔듯이 input값을 1-channel로 한 것이 아니라, 2-channel인 모델. 둘 다 word2vec으로 학습한 단어 벡터인데 하나는 static하게 값이 그대로이고 나머지 하나는 학습 중간 계속 update된다. 즉 위의 static과 non-static을 섞어서 사용한 것과 같다.



우리는 multichannel을 제외하고 구현을 시도했다.
