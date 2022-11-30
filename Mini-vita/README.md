---
title: "chapter 1 문제 풀이"
date: 2022-12-01
categories: Machine Learning
mathjax: true
writer: Mini-vita
---

$$\sum_{n=1}^N\{ y(x_n, w) - t_n \} ^2$$   (식 1.2)
$$\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}$$


## 1.5 
식 1.38의 정의를 이용해서 var[f(x)]가 식 1.39를 만족함을 증명하라
\[ [\left\{(a+b)+c\right\}+d]+e = [{(a+b)+c}+d]+e \]
$$var[f] = E[(f(x) - E[f(x)])^2] $$
$$       = E[(f(x)^2 - 2f(x)E[f(x)] + E[f(x)]^2] $$
$$       = E[f(x)^2] - 2E[f(x)]^2 + E[f(x)]^2 $$
$$       = E[f(x)^2] - E[f(x)]^2 $$



## 1.6 ## 
두 변수 x와 y가 서로 독립적일 때, x와 y의 공분산이 0임을 증명하라
$$cov[x, y] = E_{x,y} [\lbrace x - E[x]\rbrace \lbrace y-E[y]\rbrace] $$
$$          = E_{x,y}[xy] - E[x]E[y] = 0 $$

즉, 우리는 x와 y가 독립적일 때 아래와 같다는 것을 증명하면 됨!
$$ E_{x,y}[xy] = E[x]E[y] $$

x, y가 독립적이다라는 것은 
$$p(x, y) = p_x(x)p_y(y)$$


