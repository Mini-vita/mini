---
title: "chapter 1 문제 풀이"
date: 2022-12-01
categories: Machine Learning
mathjax: true
writer: Mini-vita
---

\{ \}

## 1.5 **
식 1.38의 정의를 이용해서 var[f(x)]가 식 1.39를 만족함을 증명하라
\e{x^2}
$$var[f] = E[(f(x) - E[f(x)])^2]$$
$$       = E[(f(x)^2 - 2f(x)E[f(x)] + E[f(x)]^2] $$
$$       = E[f(x)^2] - 2E[f(x)]^2 + E[f(x)]^2 $$
$$       = E[f(x)^2] - E[f(x)]^2 $$

$$\sum_{n=1}^N\{ y(x_n, w) - t_n \} ^2$$   (식 1.2)
$$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$

## 1.6 ## 
두 변수 x와 y가 서로 독립적일 때, x와 y의 공분산이 0임을 증명하라

