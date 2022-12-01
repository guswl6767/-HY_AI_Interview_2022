# HY_AI_Interview_2022

### 한양대학교 인공지능융합대학원 면접준비
----
## Statistic / Probability
🧐 Central Limit Theorem 이란 무엇인가?
```
데이터의 크기(n)가 일정한 양을 넘으면, 평균의 분포는 정규분포에 근사하게 되며, 표준편차는 모집단의 표준편차를 표본수의 제곱근으로 나눈 값과 근사.
즉 모집단으로부터 무작위로 표본을 여러 번 추출한 다음, 추출된 각각의 표본들의 평균을 분포로 그려보면 정규분포의 형태를 가짐.
주의해야할 점은 표본의 양이 충분하면 표본의 평균이 모집단의 평균과 유사해진다는 뜻이 아니라 표본을 여러 번 추출 했을 때 각각의 표본 평균들의 분포가 정규분포를 이룸.
```
[링크] : https://blog.naver.com/PostView.naver?blogId=angryking&logNo=222414551159&parentCategoryNo=&categoryNo=22&viewDate=&isShowPopularPosts=true&from=search

🧐 Central Limit Theorem은 어디에 쓸 수 있는가?
```
중심극한정리는 통계학에 있어 추정과 가설검정을 위한 핵심적인 이론으로 가설검정에 사용됨.
더 나아가 데이터 과학을 위한 예측 모델링 가능
```
🧐 큰수의 법칙이란?
```
경험적 확률과 수학적 확률 사이의 관계를 나타내는 법칙으로 표본집단의 크기가 커지면 그 표본평균이 모평균에 가까워짐을 의미. 
따라서 취합하는 표본의 수가 많을수록 통계적 정확도가 올라감
```
[링크] : https://namu.wiki/w/%ED%81%B0%20%EC%88%98%EC%9D%98%20%EB%B2%95%EC%B9%99?__cf_chl_tk=QK.VtsHZHCpJfNb.mzrLeeskXzahWBKOp5M9paBlyAg-1669796077-0-gaNycGzNCdE

🧐 확률이랑 통계랑 다른 점은?
```
확률은 어떤 사건이 일어날 수 있는 수학적기대치
확률 = 특정사건이 일어날 개수 / 전체 사건이 일어날 개수
통계는 이미 발생한 사건이나 앞으로 발생될 사건에 대해서 수준파악, 예측자료로 사용할 데이터 분석 과정으로 반복횟수가 한번이 아닌 여러 번
```
🧐 Marginal Distribution이란 무엇인가?
```
개별사건의 확률이지만 결합사건들의 합으로 표시될 수 있는 확률
X=0으로 고정할 때 P(X=0,Y=0)+P(X=0,Y=1)=P(X=0) 도출될때 X는 고정되었지만 Y의 값은 계속 변함.
즉 다시말해서 Y=y의 값에 관계없이 X=0인 주변확률이라고 표현할 수있음.
```
<img src='https://user-images.githubusercontent.com/79496166/204947444-1d465c80-a2e6-4c28-a826-1eaa4c6ce689.png'/>

🧐 Conditional Distribution이란 무엇인가?
```
조건부확률은 특정한 주어진 조건 하에서 어떤 사건이 발생할 확률을 의미.
즉 어떤사건 A가 일어났다는 전제 하에서 사건 B가 발생할 확률.
조건부확률을 어떠한 사건 A가 일어났다는 전제 하에 확률을 정의하므로 이때의 표본공간은 A의 근원사건 K개로 이루어진 표본공간으로 재정의
공식 : P(B|A) = P(A∩B)/P(A)
```
<img src='https://user-images.githubusercontent.com/79496166/204947774-f9d4fef8-57c8-4832-9c66-a83b1af457a3.png'/>
[링크] : https://datalabbit.tistory.com/17#recentComments

🧐 Bias란 무엇인가?
```
모델을 통해 얻은 예측값과 실제 정답과의 차이의 평균.
즉 예측값이 실제 정답값과 얼만큼 떨어져 있는지 나타냄.
만약 bias가 높다고 하면 그만큼 예측값과 정답값 간의 차이가 큼
Bias가 높은 게 좋을 수도 있음. 
예를 들면 통계청이 발표한 20대 남성 평균 신장은 174.21. A지역에서 평균을 내보니 175이고 B지역에서는 173이였다면, 평균은 174로, Bias는 0.21. 
그런데 C와 D지역에서는 각각 176과 172였고 이 경우에도 평균은 174로, Bias는 0.21로 동일. 
결국 Bias 안에는 평균의 함정이 숨어있음. 파라미터를 추정했을 때, 추정된 파라미터끼리 차이가 클수도 있고 작을수도 있다는 것

```
🧐 Biased/Unbiased estimation의 차이는? 
```
Unbiased Estimator는 파라미터 추정 평균에 대해서 bias값이 0인 경우
Biased Estimator는 파라미터 추정 평균의 bias 값이 0이 아닌 경우

```

🧐 Variance, MSE란?
##### variance 
```
Variance는 다양한 데이터 셋에 대하여 예측값이 얼만큼 변화할 수 있는지에 대한 양의 개념. 
이는 모델이 얼만큼 유동성을 가지는 지에 대한 의미로도 사용되며 분산의 본래 의미와 같이 얼만큼 예측값이 퍼져서 다양하게 출력될 수 있는 정도로 해석할 수있음.
```
<img src='https://user-images.githubusercontent.com/79496166/204948284-82eb5026-3788-4776-8374-a089715ae674.png'/>
[링크] : https://gaussian37.github.io/machine-learning-concept-bias_and_variance/

##### MSE
```
MSE는 오차의 제곱에 대한 평균을 취한 값으로 통계적 추정의 정확성에 대한 질적인 척도로 많이 사용됨
실제값(관측값)과 추정값의 차이로, 잔차가 얼마인지 알려주는데 많이 사용되는 척도이다.
MSE가 작을수록 추정의 정확성이 높아짐.
```

🧐 Sample Variance란 무엇인가?
```
모집단으로부터 무작위로 n개의 표본을 추출했을 때 이 n개 표본들의 평균과 분산을 각각 표본평균, 표본분산이라고 함.
```

🧐 Confidence Interval이란 무엇인가?
```
신뢰구간은 모수가 실제로 포함될 것으로 예측되는 범위
집단 전체를 연구하는 것을 불가능하므로 샘플링된 데이터를 기반으로 모수의 범위를 추정하기 위해 사용됨 
따라서 신뢰구간은 샘플링된 표본이 연구중인 모집단을 얼마나 잘 대표하는 지 측정하는 방법.
일반적으로 95% 신뢰수준이 사용됨

```
<img src='https://user-images.githubusercontent.com/79496166/204949883-d995dc66-efc4-41eb-9b02-cfc0ee3639ec.png'/>

🧐 covariance/correlation 이란 무엇인가?
```
Covariance 즉 공분산은 서로 다른 변수들 사이에 얼마나 의존하는지를 수치적으로 표현하며 그것의 직관적 의미는 어떤 변수(X)가 평균으로부터 증가 또는 감소라는 경향을 보일 때 이러한 경향을 다른 변수(Y 또는 Z 등등)가 따라하는 정도를 수치화 한 것
공분산은 또한 두 변수의 선형관계의 관련성을 측정한다 라고도 할 수 있다.

```
[링크] : https://blog.naver.com/sw4r/221025662499

```
Correlation 즉 상관관계는 상관 분석과 연관됨. 즉 두 변수 간에 어떤 선형적 관계를 갖고 있는 지를 분석하는 방법이 상관 분석인데, 두 변수는 서로 독립적인 관계이거나 상관된 관계일 수 있으며 이 때 두 변수 간의 관계의 강도를 상관관계라고 한다
```
[링크] : https://otexts.com/fppkr/causality.html

🧐 Total variation 이란 무엇인가?
```
전체분산은 각 표본의 측정치들이 전체 평균으로부터 얼마나 분산되어 있는지를 측정한 것
```
[링크] : https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=lllmanilll&logNo=140184184273

🧐 Explained variation 이란 무엇인가?
```
설명된 분산은 통계에서 주어진 데이터의 분산을 설명하는 비율을 측정함. 그중에서 설명된 분산의 비율은 전체 고윳값중에서 원하는 고윳값의 비율임.
```
[링크] : https://dnai-deny.tistory.com/16

🧐 Coefficient of determination 이란? (결정계수) r2
```
결정계수란 y의 변화가 x에 의해 몇 % 설명되는지 보여주는 값임. 예를 들면 결정계수가 0.52이면 y의 변화는 x에 의해 52% 설명된다는 뜻이다.
R의 값은 +가 될 수 도있고 -가 될 수도 있지만 r을 제곱하면 무조건 양수이므로 양의 상관관계든 음의 상관관계든 y의 변화가 x에 의해 몇 % 영향을 미친 것인지 설명이 가능함.
결정계수의 범위는 0과 1이며 만약 모든 관찰점이 회귀선상에 위치한다면 결정계수의 값은 1이 됨. 반대로 회귀선에서의 변수들 간 회귀관계가 전혀 없어 추정된 회귀선의 기울기가 0이면 결정계수의 값은 0이 된다.

```
[링크] : https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=xodh16&logNo=220545881424

🧐 P-value란 무엇인가?
```
Probability-value의 줄임말로 확률 값을 뜻하며 어떤 사건이 우연히 발생할 확률 
P-value를 어느곳에 적용하는지 알기 위해서는 가설검정에 대한 이해가 선행되어야 함.
가설검정이란 대상집단에 대하여 어떤 가설을 설정하고 검토하는 통계적 추론을 의미함.

```
<img src='https://user-images.githubusercontent.com/79496166/204950540-18263cd2-a649-4092-90f0-1befdf975883.png'/>

```
예를 들어 A회사 B회사로부터 각각 200명씩 표본을 추출하여 고객만족도 조사를 한 결과 아래와 같은 결과가 나타났다고 가정.
A회사 : 80점
B회사 : 90점
그렇다면 A회사의 표본 평균 점수인 80보다 B회사의 표본 평균 점수인 90점이 높은데 실제로 A회사의 모평균인 μa < B회사의 모평균인 μb 인지 아니면 표본의 점수일 뿐 모평균의 점수는 차이가 없는지를 가설을 설정하고 검정해야함
여기서 확인하고 싶은 것은 실제 A회사의 모집단 점수보다 B회사의 점수가 높은가? 이며 확인하고 싶은 부분을 귀무가설과 대립가설로 정의함.

```
<img src='https://user-images.githubusercontent.com/79496166/204950613-79562e92-7ff1-4794-b09d-987024f1f8fd.png'/>

```
이러한 귀무가설을 기각할 수 있는지에 대한 여부 판단 방법 중하나로 P-value(유의확률)를 확인하는 방법이 있음.
P-value의 값이 0.05보다 작다는 것은 어떤 사건이 우연히 일어날 확률이 0.05보다 작다라는 의미이며 우연히 발생할 확률이 5%보다 작다는 것은 이 사건이 우연히 일어났을 가능성이 거의 없다는 것으로 추정 가능.

```
[링크] : https://bodi.tistory.com/entry/%EA%B0%80%EC%84%A4%EA%B2%80%EC%A0%95-P-value%EB%9E%80%EC%96%B4%EB%96%A4-%EC%82%AC%EA%B1%B4%EC%9D%B4-%EC%9A%B0%EC%97%B0%ED%9E%88-%EB%B0%9C%EC%83%9D%ED%95%A0-%ED%99%95%EB%A5%A0

🧐 likelihood-ratio test 이란 무엇인가?
```
우도 비율 검정은 모형 두 개의 우도의 비를 계산해서 두 모형의 우도가 유의하게 차이나는지 비교하는 방법
```
<img src='https://user-images.githubusercontent.com/79496166/204950906-0fdaa8f4-b750-414e-9f4e-7f04c24db5fb.png'/>

```
위의 그림에서 고혈압과 당뇨가 이미 포함된 모형 A에 비만이라는 독립변수를 추가하여 모형 B를 세팅했을 때, 이 두 모형이 통계적으로 유의한 우도의 차를 보인다면 비만은 의미있는 독립변수라고 할 수 있음.
여기서 우도란 어떤 값이 관측되었을 때 해당 관측값이 어떤 확률분포로 나왔는지에 대한 확률

```
<img src='https://user-images.githubusercontent.com/79496166/204951197-2ec69764-936e-411f-8c87-46e6afe734a1.png'/>
[링크] : 	https://data-scientist-brian-kim.tistory.com/91
----

## Machine Learning

🧐 Frequentist 와 Bayesian의 차이는 무엇인가?
```
보통 통계학에서 한 사건이 장기적으로 일어날 때 발생하는 빈도를 확률이라고 하는데 확률을 사건의 빈도로 보는 것을 빈도주의(Frequentist)라고 하고 확률을 사건 발생에 대한 믿음 또는 척도로 바라보는 관점이 베이지안이라고 한다.
빈도주의와 베이지안은 확률을 해석하는 관점의 차이라고 설명할 수 있다.
빈도주의에서 빈도론자들은 얼만큼 빈번하게 특정한 사건이 반복되어 발생하는 가를 관찰하고 가설을 세우고 모델을 만들어서 검증한다. 
베이지안론자들은 고정된 데이터의 관점에서 파라미터에 대한 신념의 변화를 분석하고 확률은 사건 발생에 대한 믿음 또는 척도라고 봄.

```
<img src='https://user-images.githubusercontent.com/79496166/204951354-d78d6816-8ed9-4db1-ab9b-6f98ade96406.png'/>

🧐 Frequentist 와 Bayesian의 장점은 무엇인가?
```
빈도주의는 여러 번의 실험, 관찰을 통해 알게된 사건의 확률을 검정하므로 사건이 독립적이고 반복적이며 정규 분포형태일 때 사용하는 것이 좋다.
또한 대용량 데이터를 처리 할 수 있다면 계산이 비교적 복잡하지 않기 때문에 쉽게 처리가 가능하다.
베이지안은 확률 모델이 명확히 설정되어 있다면 조건부로 가설을 검증하기 때문에 가설의 타당성이 높아짐

```
[링크] : https://bodi.tistory.com/entry/%EA%B0%80%EC%84%A4%EA%B2%80%EC%A0%95-P-value%EB%9E%80%EC%96%B4%EB%96%A4-%EC%82%AC%EA%B1%B4%EC%9D%B4-%EC%9A%B0%EC%97%B0%ED%9E%88-%EB%B0%9C%EC%83%9D%ED%95%A0-%ED%99%95%EB%A5%A0

🧐 차원의 저주란?
```
차원의 저주란 차원이 증가하면서 학습데이터 수가 차원 수보다 적어져서 성능이 저하되는 현상을 일컫는다. 차원이 증가할수록 변수가 증가하고 개별 차원 내에서 학습할 데이터 수가 적어진다.
이때 주의할 점은 변수가 증가한다고 반드시 차원의 저주가 발생하는 것은 아니다. 관측치보다 변수 수가 많아지는 경우에 차원의 저주문제가 발생함

```
<img src='https://user-images.githubusercontent.com/79496166/204951528-e02fbecf-5a6c-4e87-ac12-66463151d0d6.png'/>

```
위 그림에서 보는 것과 같이 차원이 증가할수록 빈 공간이 많아진다.
같은 데이터지만 1차원에서는 데이터 밀도가 촘촘했던 것이 2차원, 3차원으로 차원이 커질수록 점점 데이터 간 거리가 멀어진다. 
이렇게 차원이 증가하면 빈 공간이 생기는데 빈 공간은 컴퓨터에서 0으로 채워진 공간이다. 
즉 정보가 없는 공간이기 때문에 빈 공간이 많을수록 학습 시켰을 때 모델 성능이 저하될 수 밖에 없다.

```
[링크] : https://for-my-wealthy-life.tistory.com/40

🧐 Train, Valid, Test를 나누는 이유는 무엇인가?
```
Train data는 training과정에서 학습을 하기 위한 용도로 사용된다.
validation data는 training과정에서 사용되며 학습을 하는 과정에서 중간평가를 하기 위한 용도이며 train data에서 일부를 떼내서 가져옴
test data는 training 과정이 끝난 후 성능평가를 하기 위해 사용하며 훈련한 모델을 한번도 보지 못한 데이터를 이용해서 평가를 하기 위한 용도이다.
보통 일반적으로 train:validation:test는 6:2:2로 하며 train loss는 낮은데 test loss가 높으면 훈련 데이터에 과대적합(overfitting)이 되었다는 의미이다

```
[링크] : https://velog.io/@hya0906/2022.03.03-ML-Testvalidtest%EB%82%98%EB%88%84%EB%8A%94-%EC%9D%B4%EC%9C%A0

🧐 Cross Validation이란?
```
교차검증은 보통 train set으로 모델을 훈련하고 test set으로 모델을 검증함.
그러나 고정된 test set을 통해 모델의 성능을 검증하고 수정하는 과정을 반복하면 결국 내가 만든 모델은 test set에만 잘 동작하는 모델이 된다.
즉 test set에 과적합(overfitting)하게 되므로 다른 실제 데이터를 가져와서 예측을 수행하면 엉망인 결과가 나와버리게 된다.
이를 해결하고자 하는 것이 바로 교차검증(cross validation)이다.
교차검증은 train set을 train set+ validation set으로 분리한 뒤, validation set을 사용해 검증하는 방식이다.
교차검증 기법에는 K-Fold 기법이 있음.
K-Fold는 가장 일반적으로 사용되는 교차 검증 방법이다.
보통 회귀 모델에 사용되며, 데이터가 독립적이고 동일한 분포를 가진 경우에 사용된다.
자세한 K-Fold 교차 검증 과정은 다음과 같다
```
1. 전체 데이터셋을 Training Set과 Test Set으로 나눈다.
2. Training Set를 Traing Set + Validation Set으로 사용하기 위해 k개의 폴드로 나눈다.
3. 첫 번째 폴드를 Validation Set으로 사용하고 나머지 폴드들을 Training Set으로 사용한다.
4. 모델을 Training한 뒤, 첫 번 째 Validation Set으로 평가한다.
5. 차례대로 다음 폴드를 Validation Set으로 사용하며 3번을 반복한다.
6. 총 k 개의 성능 결과가 나오며, 이 k개의 평균을 해당 학습 모델의 성능이라고 한다.
<img src='https://user-images.githubusercontent.com/79496166/204952025-3650e0bd-d8ac-4a8b-ae8f-fd848318fb19.png'/>
[링크] : https://wooono.tistory.com/105

🧐 (Super-, Unsuper-, Semi-Super) vised learning이란 무엇인가?

##### Supervised Learning
```
지도학습은 답(레이블이 달린)이 있는 데이터로 학습하는 것으로 입력값이 주어지면 입력값에 대한 label[y data]를 주어 학습 시키는 것
지도학습에는 크게 분류와 회귀가 있음
분류는 이진 분류 즉 True, False로 분류하는 것이며 다중분류는 여러값으로 분류하는 것
회귀는 어떤 데이터들의 특징을 토대로 값을 예측하는 것이다. 결과 값은 실수 값을 가짐.

```

##### Unsupervised Learning
```
비지도 학습은 정답을 따로 알려주지 않고 비슷한 데이터 들을 군집화하는 것이다. 일종의 그룹핑 알고리즘으로 볼 수 있다.
라벨링 되어있지 않은 데이터로부터 패턴이나 형태를 찾아야 하기 때문에 지도학습보다는 조금 더 난이도가 있다. 
실제로 지도 학습에서 적절한 피처를 찾아내기 위한 전처리 방법으로 비지도 학습을 이용하기도 한다.
대표적인 종류는 클러스터링, Dimentiionality Reduction, Hidden Markov Model 등을 사용한다.

```
<img src='https://user-images.githubusercontent.com/79496166/204952684-e6a706f7-195d-4214-845e-787b79f891c2.png'/>

##### Semi-Supervised Learning
```
ㄴㅇㄹ
```
##### 강화학습
```
강화학습은 분류할 수 있는 데이터가 존재하지 않고 데이터가 있어도 정답이 따로 정해져 있지 않으며 자신이 한 행동에 대해 보상을 받으며 학습하는 것을 말함.
게임을 예로들면 게임의 규칙을 따로 입력하지 않고 자신이 게임 환경에서 현재 상태에서 높은 점수를 얻는 방법을 찾아가며 행동하는 학습 방법으로 특정 학습 횟수를 초과하면 높은 점수를 획득할 수 있는 전략이 형성되게 됨.

```
[링크] : https://bangu4.tistory.com/96

🧐 Receiver Operating Characteristic Curve란 무엇인가?
```
ROC는 FPR(False positive rate)과 TPR(True Positive Rate)을 각각 x, y축으로 놓은 그래프이다.
TPR(True Positive Rate)는 1인 케이스에 대해 1로 바르게 예측하는 비율(Sensitivity)로 암 환자에 대해 암이라고 진단하는 경우를 뜻함.
FPR(False positive rate)는 0인 케이스에 대해 1로 틀리게 예측하는 비율(1-Specificity)로 정상에 대해 암이라고 진단하는 경우를 뜻함
ROC curve는 모델의 판단 기준을 연속적으로 바꾸면서 측정했을 때 FPR 과 TPR 의 변화를 나타낸 것으로 (0,0)과 (1,1)을 잇는 곡선이다.
ROC curve는 어떤 모델이 좋은 성능을 보이는 지 판단할 때 사용할 수 있다. 
즉 높은 sensitivitiy와 높은 specifity를 보이는 모델을 고르기 위해 다양한 모델에 대해 ROC curve를 그릴 때 좌상단으로 가장 많이 치우친 그래프를 갖는 모델이 가장 높은 성능을 보인다고 말할 수 있다.
```
<img src='https://user-images.githubusercontent.com/79496166/204953013-9b6f4216-b6e0-4c67-a25c-8206e8c50106.png'/>
[링크] : https://bioinfoblog.tistory.com/221

🧐 Accuracy,  Recall, Precision, f1-score에 대해서 설명해보라
##### Accuracy
```
Accuracy는 올바르게 예측된 데이터의 수를 전체 데이터의 수로 나눈 값

Accuracy는 데이터에 따라 매우 잘못된 통계를 나타낼 수 도있음.
예를들어, 내일 눈이 내릴지 아닐지를 예측하는 모델이 있다고 가정해볼때 
항상 False로 예측하는 모델의 경우 눈이 내리는 날은 그리 많지 않기떄문에  굉장히 높은 Accuracy를 가짐. 
높은 정확도를 가짐에도 해당모델은 쓸모 없음.
```
<img src='https://user-images.githubusercontent.com/79496166/204954875-75bfc367-8ae8-42f0-bc21-7fc687a90072.png'/>
<accuracy 수식>
<img src='https://user-images.githubusercontent.com/79496166/204954975-f777583e-87b4-4342-afb0-f19c2d63c2de.png'/>

##### Recall
```
Accuracy 문제를 해결하기 위해 재현율 사용.
Recall 즉 재현율은 실제로 True인 데이터를 모델이 True라고 인식한 데이터수.
만약 항상 False로 예측하는 모델의 경우는 Recall은 0이 됨

그러나 recall도 완벽하지는 않음.
예를들어 눈내림 예측기에서 항상 True라고 예측할 경우 accuracy는 낮겠지만 모델이 모든 날을 눈이 내릴 것이라
예측하기 때문에 recall은 1이됨.
해당 모델은 recall이 1이지만 쓸모 없는 모델임.
```
<img src='https://user-images.githubusercontent.com/79496166/204956193-9b0d43ce-338d-4957-a50c-f36019118593.png'/>
<recall 수식>

##### Precision
```
recall의 문제를 해결하기위해 Precision 사용.
Precision은 모델이 True로 예측한 데이터 중 실제로 True인 데이터의 수이다.
예시로 Precision은 실제로 눈이 내린 날의 수를 모델이 눈이 내릴거라 예측한 날의 수로 나눈 값이다.
```
<img src='https://user-images.githubusercontent.com/79496166/204957228-70351214-0910-4c2d-823e-ffa67aed7539.png'/>
<Precision 수식>
**Note: Precision과 recall은 서로 trade-off되는 관계가 있음.**

##### F1 score 
```
모델의 성능이 얼마나 효과적인지 설명할 수 있는 지표.
F1score는 Precision과 recall의 조화평균이다.
F1score는 Precision과 recall을 조합하여 하나의 통계치를 반환한다.
여기서 일반적인 평균이 아닌 조화 평균을 계산하였는데 그 이유는 Precision과 recall이 0에 가까울수록 F1score도 동일하게 낮은 값을 갖도록 하기 위함 이다.

예를들면 recall =1 이고 precision = 0.01로 측정된 모델은 Precision이 매우 낮기때문에 F1score에도 영향을 미치게 된다.
만약 일반적인 평균을 구하게 된다면 다음과 같다.
```
<img src='https://user-images.githubusercontent.com/79496166/204957659-2c0c1dc4-7af3-4854-b6e0-ce142fa1cd8d.png'/>

```
일반적으로 평균을 계산하면 높은 값이 나옴. 그러나 조화평균으로 계산하면 다음과 같은 결과를 얻음.
```
<img src='https://user-images.githubusercontent.com/79496166/204958000-120cbeee-7dc5-4fc7-94ea-8b564fe24c4f.png'/>

```
F1score가 매우 낮게 계산된 것이 확인됨.
```
[링크] : https://eunsukimme.github.io/ml/2019/10/21/Accuracy-Recall-Precision-F1-score/

🧐 Precision Recall Curve란 무엇인가?
```
Precision-Recall Curves는 Parameter인 Threshold를 변화시키면서 Precision과 Recall을 Plot 한 Curve. Precision-Recall Curves는 X축으로는 Recall을, Y축으로는 Precision을 가짐. 
Precision-Recall Curve는 단조함수가 아니기 때문에 이러한 이유로 ROC Curve보다 직관적이지 못하다는 단점을 가짐.

단조함수 ? 주어진 순서를 보존하는 함수
```
[링크] : https://ardentdays.tistory.com/20

🧐 Type 1 Error 와 Type 2 Error는?

```
가설검정 이론에서 1종 오류와 2종오류는 각각 귀무가설을 잘못 기각하는 오류와 귀무가설을 잘못 채택하는 오류이다.

```
##### 1종오류
```
귀무가설이 실제로 참이지만 이에 불구하고 귀무가설을 기각하는 오류이다. 
즉 실제 음성인 것을 양성으로 판정하는 경우이다. 거짓 양성 또는 알파 오류라고도 한다.

예를들면 아파트에 불이 나지 않았음에도 화재 경보 알람이 울린 경우를 말하게 된다.

```
##### 2종오류
```
귀무가설이 거짓인데도 기각하지 않아서 생기는 오류 
즉 실제 양성인 것을 음성으로 판정하는 경우이다. 

예를들면 아파트에 불이 났음에도 화재경보 알람이 울리지 않고 그대로 지나간 경우를 말하게 된다.
```
[링크] : https://angeloyeo.github.io/2021/01/26/types_of_errors.html
