# -HY_AI_Interview_2022

## 한양대학교 인공지능융합대학원 면접준비
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
🧐 Conditional Distribution이란 무엇인가?
🧐 Bias란 무엇인가? [Answer Post]
🧐 Biased/Unbiased estimation의 차이는? [Answer Post]
🧐 Bias, Variance, MSE란? 그리고 그들의 관계는 무엇인가?
🧐 Sample Variance란 무엇인가?
🧐 Variance를 구할 때, N대신에 N-1로 나눠주는 이유는 무엇인가?
🧐 Gaussian Distribution에서 MLE와 Sample Variance 중에 어떤 걸 사용해야 하는가?
🧐 Unbiased Estimation은 무조건 좋은가?
🧐 Unbiaed Estimation의 장점은 무엇인가?
🧐 Binomial, Bernoulli, Multinomial, Multinoulli 란 무엇인가?
🧐 Beta Distribution과 Dirichlet Distribution이란 무엇인가?
🧐 Gamma Distribution은 어디에 쓰이는가?
🧐 Possion distribution은 어디에 쓰이는가?
🧐 Bias and Varaince Trade-Off 란 무엇인가? [Answer Post]
🧐 Conjugate Prior란?
🧐 Confidence Interval이란 무엇인가?
🧐 covariance/correlation 이란 무엇인가?
🧐 Total variation 이란 무엇인가?
🧐 Explained variation 이란 무엇인가?
🧐 Uexplained variation 이란 무엇인가
🧐 Coefficient of determination 이란? (결정계수)
🧐 Total variation distance이란 무엇인가?
🧐 P-value란 무엇인가?
🧐 likelihood-ratio test 이란 무엇인가?

----

## Machine Learning
🧐 Frequentist 와 Bayesian의 차이는 무엇인가?
🧐 Frequentist 와 Bayesian의 장점은 무엇인가?
🧐 차원의 저주란?
🧐 Train, Valid, Test를 나누는 이유는 무엇인가?
🧐 Cross Validation이란?
🧐 (Super-, Unsuper-, Semi-Super) vised learning이란 무엇인가?
Supervised Learning
Unsupervised Learning
Semi-Supervised Learning
🧐 Decision Theory란?
🧐 Receiver Operating Characteristic Curve란 무엇인가?
🧐 Precision Recall에 대해서 설명해보라
🧐 Precision Recall Curve란 무엇인가?
🧐 Type 1 Error 와 Type 2 Error는?
