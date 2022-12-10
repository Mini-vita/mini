$$x = \begin{pmatrix} 
         x_{a} & a_{12} & a_{13}  \\
         a_{21} & a_{22} & a_{23}  \\
         a_{31} & a_{32} & a_{33}  \\
       \end{pmatrix} $$
       
## 2.3.1 조건부 가우시안 분포

만약 두 변수 집합이 결합적으로 가우시안 분포를 보인다면 \
하나의 변수 집합에 대한 다른 변수 집합의 조건부 분포 역시 가우시안 분포를 보인다는 성질을 증명하는 단원

우선 D차원의 벡터 x가 가우시안 분포라고 가정할 때, $x_a$와 $x_b$로 나누어 보자. 
 
 $$x = \begin{pmatrix} 
         x_a  \\
         x_b       
       \end{pmatrix} $$
       
 각각 부분 집합의 평균값 벡터 $\mu$ 역시 아래 처럼 정의됨
 
 $$\mu = \begin{pmatrix} 
         \mu_a  \\
         \mu_b       
       \end{pmatrix} $$
       
 그렇다면, 정방행렬이면서 대칭행렬인 공분산 행렬 $\Sigma$는 다음처럼 주어짐 
 
$$\Sigma = \begin{pmatrix} 
           \Sigma_{aa} & \Sigma_{ab} \\
           \Sigma_{ba} & \Sigma_{bb} 
           \end{pmatrix} $$

여기서 짚고 갈 점은 공분산 행렬이 대칭행렬이라는 것이다.($\Sigma^T = \Sigma$)

그렇다면, 공분산 행렬의 역행렬인 정밀도 행렬 역시 대칭행렬이 된다. (참고로 연습문제 2.2가 이걸 증명함)
$$\Lambda = \Sigma^{-1}$$

$$\Lambda = \begin{pmatrix} 
            \Lambda_{aa} & \Lambda_{ab} \\
            \Lambda_{ba} & \Lambda_{bb} 
            \end{pmatrix} $$

-----------------------------------------------------------------------------------------------------------------------------------------------------------
++정밀도 행렬이란? 
다변량이 아니라 일변량에 대해서 생각해보았을 때, 분산값의 역수이므로 precision을 뜻한다고 이해하면 됨
해당 책에서 정밀도를 이용했을 때 더 쉽게 표현할 수 있다고 했는데 대표적으로 두 가지 경우가 존재함
1) 베이즈 정리에서 사후확률의 정밀도 행렬은 prior과 likelihood의 정밀도 행렬의 단순합으로 표현됨
2) 두 차원 i와 j가 조건적 독립일 때 ij와 ji는 0이다. 만약 많은 차원이 조건적 독립적이라면 해당 행렬은 sparse 해짐 --> 계산하기 용이해짐 //
   이건 partial correlation(x와 y는 높은 상관, y와 z는 높은 상관 --> x와 z가 y를 통해서 연결되어 있다면 y를 통제했을 때 x와 z의 상관은 0)의 idea와 매우 관련있음
-----------------------------------------------------------------------------------------------------------------------------------------------------------

자 그럼 본격적으로 증명해보자! \
우선 x가 가우시안 분포라는 가정에서부터 시작해야 하며 가우시안 분포의 지수상의 이차식 형태 2.44를 참고하자. \n
또한, $x = \begin{pmatrix}
            x_a & x_b
           \end{pmatrix}^T $
           
$$\frac{-1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) = 
                                   -\frac{1}{2} \begin{pmatrix}
                                                   x_a - \mu_a & x_b - \mu_b
                                                 \end{pmatrix}
                                                 \begin{pmatrix} 
                                                 \Lambda_{aa} & \Lambda_{ab}  \\
                                                 \Lambda_{ba} & \Lambda_{bb} 
                                                 \end{pmatrix} 
                                                 \begin{pmatrix}
                                                   x_a - \mu_a  \\
                                                   x_b - \mu_b
                                                \end{pmatrix} $$
                                                
$$ = -\frac{1}{2}[(x_a - \mu_a)^T \Lambda_{aa} (x_a - \mu_a) + 2(x_a - \mu_a)^T \Lambda_{ab} (x_b - \mu_b) + (x_b - \mu_b)^T \Lambda_{bb} (x_b - \mu_a )] $$








