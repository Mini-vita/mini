
## 2.3.1 조건부 가우시안 분포

만약 두 변수 집합이 결합적으로 가우시안 분포를 보인다면 \
하나의 변수 집합에 대한 다른 변수 집합의 조건부 분포 $p(x_a | x_b)$ 역시 가우시안 분포를 보인다는 성질을 증명하는 단원

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
조건부 분포  $p(x_a | x_b)$ 에 대해서 $x_b$를 관측값으로 고정하고 \
그 결과에 해당하는 표현식을 정규화해서 $x_a$ 에 해당하는 올바른 확률 분포를 구할 수 있다. \
우선 x가 가우시안 분포라는 가정에서부터 시작해야 하며 가우시안 분포의 지수상의 이차식 형태 2.44를 참고하자. \
또한, $x = \begin{pmatrix}
            x_a & x_b
           \end{pmatrix}^T $
           
$$\frac{-1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) = $$
$$-\frac{1}{2} \begin{pmatrix}
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
                                                
$$= -\frac{1}{2}[(x_a - \mu_a)^T \Lambda_{aa} (x_a - \mu_a) + 2(x_a - \mu_a)^T \Lambda_{ab} (x_b - \mu_b) + (x_b - \mu_b)^T \Lambda_{bb} (x_b - \mu_b )] $$
$$ -\frac{1}{2}(x_a - \mu_a)^T \Lambda_{aa} (x_a - \mu_a) - (x_a - \mu_a)^T \Lambda_{ab} (x_b - \mu_b) -\frac{1}{2}(x_b - \mu_b)^T \Lambda_{bb} (x_b - \mu_b ) $$
$x_b$가 관측값이므로, $x_a$에 대해서 이차식의 형태를 띈다. \
조건부 분포 $p(x_a | x_b)$는 가우시안 분포다. \
가우시안 분포는 평균과 공분산에 의해서 결정되므로 $p(x_a | x_b)$의 평균과 공분산을 찾아내보자.  \
\

일반적으로 평균과 공분산을 찾아내는 작업은 '제곱식의 완성'을 통해서 진행된다. \
가우시안 분포의 지수식을 2.71처럼 나타낼 수 있다는 성질을 활용해서 제곱식의 완성 문제를 풀어보자. \
$$\frac{-1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) = -\frac{1}{2} x^T \Sigma^{-1}x + x^T\Sigma^{-1}\mu + const $$

------------------------------------------------------------------------------------------------------------------------------------------------
제곱식의 완성이란?
$$x^2 - 6x + 2 = x^2 - 6x + 9 - 7 = (x - 3 )^2 - 7 $$
근데 해당 책에서는 $$ax^2 + bx + c $$의 형태를 만드는 것까지만 하면 됨

------------------------------------------------------------------------------------------------------------------------------------------------

$$-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) = $$
$$-\frac{1}{2}(x^T\Sigma^{-1}x - \mu^T\Sigma^{-1}x - x^T\Sigma^{-1}\mu + mu^T\Sigma^{-1}\mu) $$ 
여기서 $\mu^T\Sigma^{-1}x = x^T\Sigma^{-1}\mu $임을 이용하면, 
$$-\frac{1}{2}x^T\Sigma^{-1}x + x^T\Sigma^{-1}\mu - \frac{1}{2}mu^T\Sigma^{-1}\mu $$
가 된다. 그럼, 마지막 항은 x에 종속적이지 않기 때문에 const가 된다. 즉, 식 2.71이 되지! \
여기서 다시 출발하면 
$$= -\frac{1}{2}\begin{pmatrix}
                x_a  & x_b                 
                \end{pmatrix}^T
                \begin{pmatrix} 
                \Lambda_{aa} & \Lambda_{ab}  \\
                \Lambda_{ba} & \Lambda_{bb} 
                \end{pmatrix}
                \begin{pmatrix}
                 x_a   \\
                 x_b 
                \end{pmatrix} + x^T\Sigma^{-1}\mu + C $$
                
$X_b$가 고정된 관측값인 경우에 , 결합분포인 x에서의 이차항은 $-\frac{1}{2}x_a^T\Lambda_{aa}x_a$ \
이걸 2.71과 같이 봤을 때, $\Sigma_{a|b}^{-1} = \Lambda{aa}$  (식 2.73)  \
자, 즉, 조건부 분포의 공분산을 구했다. 
이번에는 위에 저걸 계산했을 때 x_a의 일차식에 해당하는 항만 뽑아보면, \
$$x_a^T[\Lambda{aa}\mu_a - \Lambda{ab}(x_b - \mu_b)]$$
이다. 

2.70에서의 일차항 계수 $\Sigma^{-1}\mu$를 같이 생각해보면, 
$$\Sigma_{a|b}^{-1}\mu_{a|b} = \Lambda{aa}\mu_a - \Lambda{ab}(x_b - \mu_b) $$
즉, 
$$\mu_{a|b} = \Sigma_{a|b}[\Lambda{aa}\mu_a - \Lambda{ab}(x_b - \mu_b)] $$
$$= \mu_a - \Lambda_{aa}^{-1}\Lambda_{ab}(x_b - \mu_b) (식 2.75) $$ 

식 2.73과 2.75는 분할 정밀 행렬에 대한 식으로 표현되었다. \
이는 분할 공분산 행렬의 식으로도 표현 가능하다. 이 때, 분할 행렬의 역행렬에 대한 성질을 활용해야 한다. 

$$\begin{pmatrix}    
    A & B  \\ 
    C & D 
    \end{pmatrix} = \begin{pmatrix}
                     M         & -MBD^{-1}    \\
                     -D^{-1}CM & D^{-1}CMBD^{-1} 
                    \end{pmatrix}  (식 2.76) $$
                    
여기서 M은 다음과 같이 정의되었다. 
$$M = (A-BD^{-1}C)^{-1}  (식 2.77)$$
이걸 적용해서 생각해보면, 
$$\begin{pmatrix} 
  \Sigma_{aa} & \Sigma_{ab} \\
  \Sigma_{ba} & \Sigma_{bb} 
  \end{pmatrix}^{-1}  = \begin{pmatrix} 
                           \Lambda_{aa} & \Lambda_{ab} \\
                           \Lambda_{ba} & \Lambda_{bb} 
                        \end{pmatrix} $$
                        
여기에 식 2.76을 적용하면 다음과 같다. 
$$\Lambda_{aa} = (\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1} (식 2.79)$$
$$\Lambda_{ab} = -(\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1} \Sigma_{ab}\Sigma_{bb}^{-1} (식 2.80)$$

이 식들을 바탕으로 정밀도 행렬로 정리했던 조건부 분포의 평균과 공분산을 공분산 행렬에 대해서 정리해보면 다음과 같다. 
$$\mu_{a|b} = \mu_{a} + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b)  (식 2.81)$$
$$\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}  (식 2.82)$$

식 2.73과 2.82를 비교해보면, 조건부 분포를 표현할 때 정밀 행렬이 더 simple 함. \
끝맺음을 해보자면, 식 2.81을 보면 조건부 분포의 평균은 $x_b$에 대한 일차식이며, 식 2.82로 주어진 공분산은 $x_b$ 에 대해 독립적이다. 
이것이 바로 선형 가우시안 모델의 예다. 


## 2.3.2 주변 가우시안 분포
해당 단원은 주변 분포 역시 가우시안 분포라는 것을 증명하는 단원 \
이제 위에서 한 식을 주변 확률 분포에 대해서도 살펴보자. 

------------------------------------------------------------------------------------------------------------------------------------------------
주변 확률 분포
1) 결합 확률 분포에서 한 쪽의 변수가 사라지거나 무시되는 것
2) p(x, y)에 대해서 x의 주변 확률 분포는 p(x)가 된다. 
3) 즉, 한 쪽의 변수가 합산하여 사라지게 되는데 이산 변수는 모든 확률 값의 합으로, 연속 변수의 경우 적분을 통해 진행된다. 

------------------------------------------------------------------------------------------------------------------------------------------------
$$p(x_a) = \int p(x_a, x_b)dx_b  (식 2.83) $$
윗 단원에서처럼 지수상의 이차식(식 2.70)에 초점을 맞춰서 주변 분포의 평균과 공분산을 구하는 전략을 사용하자. 

$$\frac{-1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) $$
$$= -\frac{1}{2}(x_a-\mu_a)^T\Lambda_{aa}(x_a - \mu_a)  -\frac{1}{2}(x_a-\mu_a)^T\Lambda_{ab}(x_b - \mu_b)
    -\frac{1}{2}(x_b-\mu_b)^T\Lambda_{ba}(x_a - \mu_a)  -\frac{1}{2}(x_b-\mu_b)^T\Lambda_{bb}(x_b - \mu_b) (식 2.70) $$
    
$$= -\frac{1}{2}(x_a^T\Lambda_{aa}x_a - x_a^T\Lambda_{aa}\mu_a - \mu_a^T\Lambda_{aa}x_a + \mu_a^T\Lambda_{aa}\mu_a) \\
 -\frac{1}{2}(x_a^T\Lambda_{ab}x_b - x_a^T\Lambda_{ab}\mu_b - \mu_a^T\Lambda_{ab}x_b + \mu_a^T\Lambda_{ab}\mu_b) \\
 -\frac{1}{2}(x_b^T\Lambda_{ba}x_a - x_b^T\Lambda_{ba}\mu_a - \mu_b^T\Lambda_{ba}x_a + \mu_b^T\Lambda_{ba}\mu_a) \\
 -\frac{1}{2}(x_b^T\Lambda_{bb}x_b - x_b^T\Lambda_{bb}\mu_b - \mu_b^T\Lambda_{bb}x_b + \mu_b^T\Lambda_{bb}\mu_b) \\ $$

위에서 $x_b$에 대한 이차식과 일차식을 뽑아내보자
(5번 = 8번, 7번 = 10번, 14번 = 15번, 그리고 13번이 이차식)
 $$= -\frac{1}{2}x_b^T\Lambda_{bb}x_b + x_b^T(\Lambda_{bb}\mu_b - \Lambda_{ba}(x_a - \mu_a)) + 나머지 \\
 = -\frac{1}{2}x_b^T\Lambda_{bb}x_b + x_b^Tm + 나머지  $$ 
 
 여기서 짚고 갈 점은 두 가지!
 우선, m을 정의할 수 있음
$$m = \Lambda_{bb}\mu_b - \Lambda_{ba}(x_a - \mu_a)  ( 2.85) $$
또한, 제곱식을 완성할 수 있다. 
$$-\frac{1}{2}x_b^T\Lambda_{bb}x_b + x_b^Tm =  -\frac{1}{2}(x_b - \Lambda_{bb}^{-1}m)^T\Lambda_{bb}(x_b - \Lambda_{bb}^{-1}m) + \frac{1}{2}m^T\Lambda_{bb}^{-1}m  (식 2.84) $$
a의 주변 확률 분포는 b에 대한 항들을 적분하면 되지! $x_b$에 대해서 제곱식 완성한 식 2.84에 지수함수를 취해서 $x_b$를 적분 시켜서 없앨 수 있다. 그렇다면, 남게 되는 것은 식 2.84의 오른쪽 항과 식 2.70에서 $x_a$에만 종속적인 나머지 항들이다.  //
식 2.84의 오른쪽 항에 위에서 정의한 m을 대입하고 식 2.70 나머지들을 합치면 다음과 같다! 

$$\frac{1}{2}[\Lambda_{bb}\mu_b - \Lambda_{ba}(x_a - \mu_a)]^T\Lambda_{bb}^{-1}[\Lambda_{bb}\mu_b - \Lambda_{ba}(x_a - \mu_a)]  \\
-\frac{1}{2}x_a^T\Lambda_{aa}x_a + x_a^T(\Lambda_{aa}\mu_a + \Lambda_{ab}\mu_b) + const  \\
= -\frac{1}{2}x_a^T(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})x_a + x_a^T(\Lambda_{aa} - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_a + const   (식 2.87) $$

자! $x_b$를 적분하여 $x_b$를 제거한 $x_a$의 주변 확률을 봤더니, $x_a$의 이차항을 보니 가우시안 분포를 보인다! 
결국 주변 분포 역시 가우시안 분포임을 증명했다. 

가우시안 분포의 평균과 분산을 구하는 과정은 위 단원과 비슷하므로 생략한다. 

참고로, 조건부 분포에 대해서는 분할 정밀 행렬을 사용했을 때 평균과 공분산이 단순하게 표현되었던 반면에 주변 분포의 경우에는 \
분할 공분산 행렬을 사용할 때, 평균과 공분산이 가장 단순하게 표현된다. 


## 2.3.3 가우시안 변수에 대한 

