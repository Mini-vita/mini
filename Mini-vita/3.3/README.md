## 3.3 베이지안 선형 회귀

$$t = y(x, w) + \epsilon (식 3.7)$$ 
$$t = w^T\phi(x) + \epsilon$$
타깃 변수 t는 결정 함수 y(x, w)와 가우시안 노이즈의 합으로 주어진다. \
또한, 가우시안 노이즈는 $$\epsilon \sim N(0, \beta^{-1})$$    \
w는 random variable 좀 더 정확하게 말하면 random vector이다. 그러므로 w는 그 자체의 분포를 가지게 된다.

prior distribution : p(w) = N(m_0, S_0) 즉, 평균 벡터와 공분산 행  \
지금까지는 우린 아무런 데이터 즉 아무런 정보가 없으므로 $m_0$으로 정의한다. \

train data: $D = {(x_n, t_n)}^N_{n=1}$

$$t1 = w^T\phi_1(x_1) + \epsilon_1 = \phi^T(x_1)w + \epsilon_1$$
$$t2 = \phi^T(x_2)w + \epsilon_2$$
$$t_N = \phi^T(x_N)w + \epsilon_N$$

이를 행렬 form으로 바꾸면!
$$\begin{pmatrix}a & b\\\ c & d\end{pmatrix}$$
$$\begin{pmatrix}t_1 \\\ \dotsb \\\ t_N\end{pmatrix} = \begin{pmatrix}\phi^T(x_1) \\\ \dotsb \\\ \phi^T(x_N)\end{pmatrix}w + \begin{pmatirx}\epsilon_1 \\\ \dotsb \\\ \epsilon_N\end{pmatrix}$$

즉 이는 벡터로 봤을 때
$$t = \Upphi w + \epsilon$$
여기서 noise 벡터인 $\epsilon$은 $N(0, \beta^{-1}I)$의 형태이다. 이는 train data는 independent and identically distributed이며 해당 data의 noise 역시 독립적이므로 diagonal임 (식 3.52)
그런데 여기서 w의 사전 분포는 가우시안임 
아래의 likelihood를 정의해보자. D는 이미 주어진 training data이며 결국 w에 대한 함수임
$$p(D|w) = N(t|\Upphi w, \beta^{-1}I)$$

자 이제 그럼 posterior distribution을 보자. 
베이지안에 따르면, $p(w|D) = p(w, D) / P(D)$인데 P(D)는 이미 주어져 있으므로 이는 $\propto P(w, D) = P(D|W)P(W)$   \
즉, 사후확률은 likelihood와 prior의 곱이다. prior은 가우시안이며 likelihood 역시 가우시안이므로 사후 확률 역시 multivariate 가우시안이 된다. 

-------------------------------------------------------------------------
Multivariate normal distribution을 다시 확인해보면, 

![image](https://user-images.githubusercontent.com/71582504/212530440-9f6710c3-552e-4ad5-9c0e-a6dee076d284.png)
$$exp(-\frac{1}{2} (x - \mu) ^T\Sigma^{-1}(x-\mu)$$
exponential 안을 보면 $\mu$와 $\Sigma$를 뽑아낼 수 있음 
저걸 풀어내면 $-\frac{1}{2}x^T\Sigma^{-1}x + x^T\Sigma^{-1}\mu + const$이다. 
첫번째를 보면, 공분산을 뽑아낼 수 있고 알아낸 공분산을 활용하면 두 번째 항에서 평균을 뽑아낼 수 있다. 
--------------------------------------------------------------------------

자 다시 돌아가서, 위에 우리가 알아낸 가능도(이것 역시 multivariate normal distribution)와 사전 함수를 넣어 버리면
$$\propto exp(-\frac{1}{2}(t - \Upphi w)(\beta^{-1}I)^{-1}(t - \Upphi w))exp(-\frac{1}{2}(W-m_0)^T S^{-1}_0 (W-m_0))$$
$$=exp








베이지안 방법론을 통해 과적합 문제를 피할 수 있으며, 훈련 데이터만 가지고 모델의 복잡도를 결정할 수 있다. 
아래의 챕터들에서는 논의를 쉽게 하기 위해 단일 타깃 변수 t의 경우에 대해서만 살펴볼 것이다. 

## 3.3.1 매개변수 분포
베이지안 정리에 따라 w의 사후분포는 가능도함수와 사전분포의 곱으로 비례할 것이다. 

가장 가능성이 높은 w를 찾는 방식으로 w를 결정할 수 있다. 즉, 사후분포를 최대화하는 방식으로 w를 결정할 수 있으며, 이런 테크닉을 "최대 사후분포(MAP)라고 한다. 
음의 로그를 취한 뒤 사후확률의 최댓값을 찾는 것은 아래 식의 최솟값을 찾는 과정과 동일하다. 

모델 매개변수 W에 대한 사전 확률 분포를 도입하자. 
$$p(w) = N(w|m_0, S_0)  (식 3.48)$$
그 다음 단계는 사후 분포를 계산하는 것으로 사전 분포와 가능도 함수의 곱에 비례함
$$p(w|t) = N(w|m_N, S_N)  (식 3.49)$$

--------------------------------------------------------------------------
$$p(t|x, X, t) = \int p(t|x, W) p(W|X, t)dw 
사전 분포가 가우시안 분포이므로 최빈값과 평균값이 같다. 
따라서, 최대 사후 가중 벡터는 단순히 w_MAP = m_N으로 주어지게 된다. 


![image](https://user-images.githubusercontent.com/71582504/211182349-ac5733b9-f330-4f34-bf5e-481f47325463.png)
