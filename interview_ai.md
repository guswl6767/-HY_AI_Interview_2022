### 한양대학교 인공지능 융합대학원 면접 준비
----

🧐 AI, ML, DL의 차이점
```
AI는 기계가 사람을 흉내낼 수 있는 기술 또는 알고리즘
ML은 전문가가 주는 데이터를 컴퓨터가 기계(모델, 함수)로 학습하는 알고리즘
DL은 ML의 일부분으로 기계(모델, 함수)가 인간의 두노의 신경세포 뉴런이 연결된 형태를 모방한 신경망 구조.
```
🧐 지도학습, 비지도학습, 강화학습 모델
### 지도학습
```
지도학습은 말 그대로 정답이 있는 데이터를 활용해 데이터를 학습시키는 것
지도학습에는 대표적으로 분류와 회귀가 있다. 
분류는 전형적인 지도 학습이며 회귀는 어떤 데이터들의 예측 변수라 불리는 특성을 기준으로 연속된 값을 예측하는 문제로 주로 어떤 패턴이나 트렌드, 경향을 예측할 때 사용된다.

분류모델에는 KNN(최근접이웃), SVM(서포트 벡터 머신), 결정트리와 랜덤포레스트, Naive Bayes(나이브 베이즈)등이 있다.
```
##### KNN
```
다양한 레이블의 데이터 중에서 자신과 가까운 데이터를 찾아 자신의 레이블을 결정하는 방식
```
##### Naive Bayes 
```
조건부확률을 기반으로 하며 데이터 특징(Feature)이 모두 동등하고 독립적이라 가정

만약 심슨에 대한 Feature에 대해 age와 sex 2개가 존재하는 상황에서, 심슨의 나이, 성별에 따라 survive할 확률은 얼마인지를 Naive Bayes의 이론에 따라 계산하게 된다.
```
<img src='https://user-images.githubusercontent.com/79496166/204980854-4ff42f50-f0ed-447b-ad55-e7869acecfcd.png'/>

##### SVM
```
서포트 벡터 머신은 기본적으로 Decision Boundary라는 직선이 주어진 상태이다.
주로 다루려는 데이터가 2개의 그룹으로 분류될 때 많이 사용된다.
학습데이터가 벡터 공간에 위치하고 있다고 생각하며 학습 데이터의 특징(feature)수를 조절함으로써 2개의 그룹을 분류하는 경계선을 찾고, 이를 기반으로 패턴을 인식하는 방법이다.

두 그룹을 분류하는 경계선은 최대한 두 그룹에서 멀리 떨어져 있는 경계선을 구하게 되며 이는 두 그룹과의 거리(margin)를 최대로 만드는 것이 나중에 입력된 데이터를 분류할 때 더 높은 정확도를 얻을 수 있기 때문이다.

즉 Margin, 두 그룹과의 거리를 최대로 하는 직선을 찾아 최적의 분류를 하는 모델.
```

##### Decision Tree
```
가장 단순한 분류 모델 중 하나로 decision tree와 같은 도구를 활용하여 모델을 그래프로 그리는 매우 단순한 구조로 되어 있음
이 방식은 root에서부터 적절한 node를 선택하면서 진행하다가 최종 결정을 내리게 되는 model.
이 트리의 장점은 누구나 쉽게 이해할 수 있고 결과를 해석할 수 있음
불순도(gini, entropy 함수)가 낮도록 진행
여기서 entropy는 불순도를 측정하는 지표로서 정보량의 기댓값.
불순한 상태 즉 entropy가 최대값을 가질수록 분류하기가 어려움. 
이에따라 entropy를 작게하는 방향으로 가지를 뻗어나가며 의사결정 나무를 키워나가는 것.
```
🧐 Eigenvector, eigenvalue란
```
고유벡터는 그 행렬이 벡터에 작용하는 주축의 방향을 나타내므로 공분산 행렬의 고유 벡터는 데이터가 어떤 방향으로 분산되어 있는지를 나타냄.

고유값은 고유벡터 방향으로 얼마만큼 크기로 벡터공간이 늘려지는 지를 얘기한다.
따라서 고유 값이 큰 순서대로 고유 벡터를 정렬하면 결과적으로 중요한 순서대로 주성분을 구하는 것이 된다.
```
[링크] : https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-19-%ED%96%89%EB%A0%AC

🧐 PCA 개념 설명
```
PCA란 데이터 집합 내에 존재하는 각 데이터의 차이를 가장 잘 나타내주는 요소,  데이터를 잘 표현할 수 있는 특성을 찾아내는 방법으로 통계데이터 분석, 데이터 압축, 노이즈제거 등 다양한 분야에서 사용되고 있다.
특히 특성이 많은 경우 특성간 상관관계가 높을 가능성도 있는데 선형 회귀와 같은 선형 모델에서는 입력한 변수들간의 상관관계가 높을 것으로 인해 다중공선성 문제로 모델의 예측 성능이 저하 될 수 있기때문에 꼭 필요한 과정임.

PCA는 서로 상관성이 높은 여러 변수들의 선형조합으로 만든 새로운 변수들로 요약 및 축약하는 기법으로 데이터의 분산을 최대한 보존하면서 직교하는 새 기저(축)을 찾아 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간으로 변환해 준다. 
간단하게 가장 높은 분산을 가지는 데이터의 축을 찾아 차원을 축소하는데 이것이 PCA의 주성분이 되는 방식이다.

요약하자면 공분산행렬에 대한 고유값 분해로 주성분 방향벡터 즉 데이터가 어떤 방향으로 분산되어 있는지를 찾고 이를 새로운 축으로 사용하여 차원축소를 하는 알고리즘.

공분산은 2개의 확률 변수의 선형관계를 나타내는 값.
```
<img src='https://user-images.githubusercontent.com/79496166/204996171-e2d157ca-2de1-4de8-9c50-d6905161718a.png'/>
[링크] : https://ddongwon.tistory.com/114
[링크] : https://angeloyeo.github.io/2019/07/27/PCA.html

🧐 Singular value decomposition이란
```
특이값 분해는 임의의 행렬을 특정한 구조로 분해하는 방식
선형대수에서 특이값 분해는 행렬분해 방식중 하나임.
이 방법은 행렬의 고유값과 고유백터를 이용한 고유분해의 일반화 된 방식.
```
🧐 MLE, MAP의 가장 큰 차이점
```
MLE 방법의 경우 말 그대로 likelihood를 최대로 하는 방법
MAP 방법은 Posterior를 최대화 하는 방법

예시로 구별하는 MLE와 MAP의 차이
전 세계에 남자는 10%밖에 없고, 여자는 90%를 차지한다. 이때, 특정한 길이의 머리카락을 주웠다.
MLE방법은 성비를 완전히 무시한 채 그저 남자에게서 해당 머리카락이 나올 확률 p(z|남)
여자에게서 해당 머리카락이 나올 확률 p(z|여)  중 큰 것을 선택한다.
반면, MAP방법은 남녀의 성비까지 고려하여 어느 모델에서 발생되었는지를 판단하는 방법이다.

즉 
ML(Maximum Likelihood) 방법: ML 방법은 남자에게서 그러한 머리카락이 나올 확률 p(z|남)과 여자에게서 그러한 머리카락이 나올 확률 p(z|여)을 비교해서 가장 확률이 큰, 즉 likelihood가 가장 큰 클래스(성별)를 선택하는 방법이다.
MAP(Maximum A Posteriori) 방법: MAP 방법은 z라는 머리카락이 발견되었는데 그것이 남자것일 확률 p(남|z), 그것이 여자것일 확률 p(여|z)를 비교해서 둘 중 큰 값을 갖는 클래스(성별)를 선택하는 방법이다. 즉, 사후확률(posterior prabability)를 최대화시키는 방법으로서 MAP에서 사후확률을 계산할 때 베이즈 정리가 이용된다.

```
[링크] : https://niceguy1575.medium.com/mle%EC%99%80-map%EC%9D%98-%EC%B0%A8%EC%9D%B4-7d2cc0bee9c

🧐 베이즈 정리
```
데이터라는 조건이 주어졌을 때의 조건부확률을 구하는 공식이다. 베이즈 정리를 쓰면 데이터가 주어지기 전의 사전확률값이 데이터가 주어지면서 어떻게 변하는지 계산할 수 있다. 따라서 데이터가 주어지기 전에 이미 어느 정도 확률값을 예측하고 있을 때 이를 새로 수집한 데이터와 합쳐서 최종 결과에 반영할 수 있다. 
 
P(A|B): 사후확률(posterior). 사건 B가 발생한 후 갱신된 사건 A의 확률
P(A): 사전확률(prior). 사건 B가 발생하기 전에 가지고 있던 사건 A의 확률
P(B|A): 가능도(likelihood). 사건 A가 발생한 경우 사건 B의 확률
P(B): 정규화 상수(normalizing constant) 또는 증거(evidence). 확률의 크기 조정
```



