## 3.3 베이지안 선형 회귀

$$t = y(x, w) + \epsilon (식 3.7)$$ 
$$t = w^T\phi(x) + \epsilon$$
타깃 변수 t는 결정 함수 y(x, w)와 가우시안 노이즈의 합으로 주어진다. \
또한, 가우시안 노이즈는 $$\epsilon \sim N(0, \beta^{-1})$$    

## 3.3.1 매개변수 분포
w는 random variable 좀 더 정확하게 말하면 random vector이다. 그러므로 w는 그 자체의 분포를 가지게 된다.

prior distribution : p(w) = N(m_0, S_0) 즉, 평균 벡터와 공분산 행  \
지금까지는 우린 아무런 데이터 즉 아무런 정보가 없으므로 $m_0$으로 정의한다. \

train data: $D = {(x_n, t_n)}^N_{n=1}$

$$t1 = w^T\phi_1(x_1) + \epsilon_1 = \phi^T(x_1)w + \epsilon_1$$
$$t2 = \phi^T(x_2)w + \epsilon_2$$
$$t_N = \phi^T(x_N)w + \epsilon_N$$

이를 행렬 form으로 바꾸면!
$$\begin{pmatrix}t_1 \\\ \dotsb \\\ t_N\end{pmatrix} = \begin{pmatrix}\phi^T(x_1) \\\ \dotsb \\\ \phi^T(x_N)\end{pmatrix}w + \begin{pmatrix}\epsilon_1 \\\ \dotsb \\\ \epsilon_N\end{pmatrix}$$

즉 이는 벡터로 봤을 때
$$t = \phi w + \epsilon$$
여기서 noise 벡터인 $\epsilon$은 $N(0, \beta^{-1}I)$의 형태이다. 이는 train data는 independent and identically distributed이며 해당 data의 noise 역시 독립적이므로 diagonal임 (식 3.52)
그런데 여기서 w의 사전 분포는 가우시안임 
아래의 likelihood를 정의해보자. D는 이미 주어진 training data이며 결국 w에 대한 함수임
$$p(D|w) = N(t|\phi w, \beta^{-1}I)$$

자 이제 그럼 posterior distribution을 보자. 
베이지안에 따르면, $p(w|D) = p(w, D) / P(D)$인데 P(D)는 이미 주어져 있으므로 이는 $\propto P(w, D) = P(D|W)P(W)$   \
즉, 사후확률은 likelihood와 prior의 곱이다. prior은 가우시안이며 likelihood 역시 가우시안이므로 사후 확률 역시 multivariate 가우시안이 된다. 

-------------------------------------------------------------------------
Multivariate normal distribution을 다시 확인해보면, 

![image](https://user-images.githubusercontent.com/71582504/212530440-9f6710c3-552e-4ad5-9c0e-a6dee076d284.png)
-------------------------------------------------------------------------

$$exp(-\frac{1}{2} (x - \mu) ^T\Sigma^{-1}(x-\mu)$$
exponential 안을 보면 $\mu$와 $\Sigma$를 뽑아낼 수 있음 
저걸 풀어내면 $-\frac{1}{2}x^T\Sigma^{-1}x + x^T\Sigma^{-1}\mu + const$이다. 
첫번째에 있는 이차항을 보면, 공분산을 뽑아낼 수 있고 알아낸 공분산을 활용하면 두 번째 차일차항에서 평균을 뽑아낼 수 있다. 


자 다시 돌아가서, 위에 우리가 알아낸 가능도(이것 역시 multivariate normal distribution)와 사전 함수를 넣어 버리면
$$\propto exp(-\frac{1}{2}(t - \phi w)^T(\beta^{-1}I)^{-1}(t - \phi w))exp(-\frac{1}{2}(W-m_0)^T S^{-1}_0 (W-m_0))$$
참고로, $(\beta^{-1}I)^{-1} = \beta I$이므로 I만 남기고 $\beta$는 앞으로 보내버림 그리고 이차항만 뽑아내보면,
$$=exp(-\frac{\beta}{2}(w^T\phi^T\phi W + W^tS^{-1}W)$$
이며, 지수족 안에는 $-\frac{1}{2}W^T(\beta\phi^T \phi + S^{-1}_0)W$로 정리됨
따라서, $S^{-1}_N = \beta\phi^T \phi + S^{-1}_0$ 이고 이는 사후분포의 공분산이다. 이는 식 3.51이다. 이제 알아낸 공분산 행렬을 넣어서 평균 벡터를 구할 수 있다. 아까 위에서 이번엔 일차항만 뽑아내보자. 

$W^T(\beta\phi^T t + S^{-1}_0 m_0)$이고 그러므로, $S^{-1}_N m_N = \beta\phi^T t + S^{-1}_0 m_0$이다. \
따라서, 사후분포의 평균 $m_N = S_N(\beta\phi^T t + S^{-1}_0 m_0)$이다. 이는 식 3.50으로 확인된다. \
결론적으로, 베이지안 방법론을 바탕으로 선형회귀를 시행하여 훈련 데이터만 가지고 매개변수의 분포의 평균과 공분산을 알아냈다. 이렇게 알아낸 것들을 통해서 몇 가지 흥미로운 시사점을 뽑아낼 수 있다. 

대부분 우리가 어떤 데이터를 관찰하기 전에는 어떠한 정보도 없으므로 사전 분포의 $m_0 = 0$이다.  그리고 $S_0 = \alpha^{-1}I$로 가정한다. $\alpha$는 precision임
$P(w|D) = N(w|m_N, S_N)$이므로, $S_N = (\beta\phi^T \phi + \alpha I)^{-1}$이며, $m_N = S_N(\beta\phi^T t) = (\beta\phi^T \phi + \alpha I)^{-1}(\beta\phi^T t)$이다. 

1) 만약에 $\alpha$가 0이라면? 즉, precision이 0이라면? covariance가 무한이겠지. 그렇다면 final result는 $S_N = \beta\phi^T \phi$가 되고, $m_N = (\phi^T \phi)^{-1}(\phi^T t)$가 된다. 즉, $m_N = (\phi^T\phi)^{-1}(\phi^T t)$가 되며 이는 least square solution이 된다.  $= w_{LS}$
즉, 우리가 아무런 정보가 없을 때 least square solution이 된다. 

마지막으로 단순한 직선 피팅 예시를 바탕으로 선형 기저 함수 모델의 베이지안 학습과 사후 분포의 순차적 업데이트 방식에 대해 살펴보자. 
![image](https://user-images.githubusercontent.com/71582504/212532785-1bbced46-1e1e-4c55-89a0-f9a9f3e0fa5e.png)



## 3.3.2 예측 분포
실제로는 W를 알아내는 것보다는 새로운 X에 대하여 t의 값을 예측하는 것이 더 중요할 수 있다. 
prior p(w} =  \
train data D =  \
posterior p(w|D) = \

$x^* $는 test input이다. 
$$p(t^* | x^* ) = \int p(t^* , w|, D)dw = p(t^* |w)p(w|D)$$
$$model: t^* = w^T\phi(x^* ) + \epsilon$$
즉, 지금 우리가 필요한 건 w, test input, epsilon이다. 
$$\int p(t^* |w, x^* , \beta)p(w|D) dw$$
여기서 오른쪽 사후분포는 가우시안 $N(w|m_N, S_N)$이고, 왼쪽도 저 model을 보면 모두 가우시안이다. 
$p(t^* |D) =$ ???

식 3.58을 보면,
$$p(t^* |D) = N(t^* | m_N^T\phi(x^* ), \sigma^2_N(x))$$
이고, 예측 분포의 분산은 다음과 같다고 한다. 
$$\sigma^2_N(x) = \frac{1}{\beta| + \phi(x)^TS_N \phi(x)$$
$$model: t^* = w^T\phi(x^* ) + \epsilon$$
을 보면, w도 가우시안이고 epsilon도 가우시안이므로 t도 가우시안 분포가 된다. 

$$ E(t^* | D) = E(w^T \phi(x^* ) + \epsilon)$$




