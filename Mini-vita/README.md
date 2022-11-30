---
title: "chapter 1 문제 풀이"
date: 2022-12-01
categories: Machine Learning
mathjax: true
writer: Mini-vita
---

$$\sum_{n=1}^N\{ y(x_n, w) - t_n \} ^2$$   (식 1.2)
$$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$


## 1.5* 
식 1.38의 정의를 이용해서 var[f(x)]가 식 1.39를 만족함을 증명하라
\[ [\left\{(a+b)+c\right\}+d]+e = [{(a+b)+c}+d]+e \]
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
$$                                          = 







