---
title: "chapter 1 문제 풀이"
date: 2022-12-01
categories: Machine Learning
mathjax: true
writer: Mini-vita
---
## 1.1*
## 1.10 *
통계적으로 독립적인 두 변수 x와 z에 대해 두 변수 합의 평균과 분산이 다음을 만족함을 증명하라 
E[x + z] = E[x] + E[z]
VAR[x + z] = VAR[x] + VAR[z]

식 1.34를 같이 보면,  
$$ \int $$
$$ $$
$$ $$


## 1.5* 
식 1.38의 정의를 이용해서 var[f(x)]가 식 1.39를 만족함을 증명하라
$$var[f] = E[(f(x) - E[f(x)])^2] $$
$$       = E[(f(x)^2 - 2f(x)E[f(x)] + E[f(x)]^2] $$
$$       = E[f(x)^2] - 2E[f(x)]^2 + E[f(x)]^2 $$
$$       = E[f(x)^2] - E[f(x)]^2 $$


## 1.6* 
두 변수 x와 y가 서로 독립적일 때, x와 y의 공분산이 0임을 증명하라
참고로, 어떤 함수 f(x)의 평균값(\mu)은 f(x)의 기댓값(E)이다.

x와 y가 독립적이면, p(xy) = p(x)p(y) 이고
공분산은
$$cov[X, Y] = E((X - \mu_X)(Y - \mu_Y)) $$
$$          = E(XY - \mu_XY - X\mu_Y + \mu_X\mu_Y) $$
$$          = E(XY) - \mu_XE(Y) - E(X)\mu_X + \mu_X\mu_Y $$
$$          = E(XY) - E(X)E(Y) $$

그래서 x, y가 독립적이면 공분산이 0이 된다. 


## 1.14 **
$$\sum_{i=1}^D\sum_{j=1}^Dw_{i,j}x_ix_j = \sum_{i=1}^D\sum_{j=1}^D(w_{ij}^S + w_{ij}^A)x_ix_j $$
$$                                      = \sum_{i=1}^D\sum_{j=1}^Dw_{ij}^Sx_ix_j + \sum_{i=1}^D\sum_{j=1}^D w_{ij}^Ax_ix_j $$

즉, 
$$ \sum_{i=1}^D\sum_{j=1}^D w_{ij}^Ax_ix_j = 0 $$
임을 증명해야 한다. (이게 비대칭 행렬로부터의 영향이 없어진다는 것임!)

그러면 저거에 2를 곱한 것도 0이 되겠지! 

$$ 2\sum_{i=1}^D\sum_{j=1}^D w_{ij}^Ax_ix_j = \sum_{i=1}^D\sum_{j=1}^D (w_{ij}^A + w_{ij}^A)x_ix_j $$
$$                                          = \sum_{i=1}^D\sum_{j=1}^D (w_{ij}^A - w_{jI}^A)x_ix_j $$
$$                                          = \sum_{i=1}^D\sum_{j=1}^D w_{ij}^Ax_ix_j - \sum_{i=1}^D\sum_{j=1}^D w_{ji}^Ax_ix_j $$
$$                                          = \sum_{i=1}^D\sum_{j=1}^D w_{ij}^Ax_ix_j - \sum_{i=1}^D\sum_{j=1}^D w_{ij}^Ax_ix_j $$
$$                                          = 0 $$

i, j에 대해서 대칭적이니깐 i에 대해서만 고민해보면
i = 1, 2, ..., D 일 때
1 + 2 +... + D = D(D+1) / 2
인걸까.....? (으아아아아악)


## 1.28 *
1.6 절에서 p(x) 분포하에서 확률 변수 x의 값을 관측했을 때 얻게 된 정보의 양으로써 엔트로피 h(x)를 소개하였다.

1)
$$ h(x) = f(p(x)) $$


2)
p(x, y) = p(x)p(y) 인 독립변수 x와 y에 대해서 엔트로피 함수들은 가산이 가능하다고 한다. 


3)
즉, h(x, y) = h(x) + h(y) 이라고 한다. 

1) 생각했을 때, 
$$ h(x, y) = f(p(x, y)) $$
$$         = f(p(x)p(y)) $$

3)과 1) 생각했을 때, 
$$ h(x, y) = h(x) + h(y) $$
$$ h(x) + h(y) = f(p(x)) + f(p(y)) $$

종합해보면,
$$ f(p(x)) + f(p(y)) = f(p(x)p(y)) $$
p(x)를 x라고 하고 p(y)를 y라고 하면 결국
$$ f(xy) = f(x) + f(y) $$

우선 첫번째로
$$ h(p^2) = 2h(p) $$ 
이다. 
그럼, 2를 포함한 모든 정수에도 이것이 적용될 것이다. 
n이 맞으면 n+1에도 적용되기 때문! 

$$ f(x^{n+1}) = f(x^n) + f(x) = nf(x) + f(x) = (n+1)f(x) $$

다음 증명으로 넘어가면
$$f(x^n) = f((x_{n/m})^m) = mf(x^{n/m}) $$
$$nf(x) = mf(x^{n/m}) $$
$$n/mf(x) = f(x^{n/m}) $$




