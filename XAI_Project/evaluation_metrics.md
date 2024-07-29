# **Evaluation Metrics**

## **Metric 1: Mean Computation Time per Instance**

$$
\text{mean per instance} = \frac{\text{time to compute local dataset}}{\text{n}}
$$

$X$ = given dataset

$n$ = # of instances

$d$ = # of features



## **Metric 2: Error with regards to the Complete method**

$$
err(f, X) = \frac{1}{n}\Sigma^n_{i=1}\frac{1}{p}\Sigma^d_{k=1}{|f_k(X_i) - f_k^C{x_I}|}
$$

$f_k(x)$ be the influence of a feature $k$ produced by an explanation method $f$ for a given instance $x$, and a given machine learning model

$f_k^C(x)$ influence given by the cComplete method for same model, same feature, and same instance

$p$ number of errors since this is calculating an average

## **Metric 3: Area under the cumulative feature importance curve**

$$
AUC(x) = \frac{1}{d}\Sigma^{d-1}_{i=0}{\frac{C_i + C_{i+1}}{2}}
$$

$C \in [0; 1]^{d+1}$: cumulative importance proportion vector given by explanation method
    - $C_i$: total importance proportion taken by the $i$ most important features


## **Metric 4: Robustness**

$$
L_X(x_i) = \max_{x_j\in N_\epsilon(x_i) \leq \epsilon} \frac{||f(x_i) - f(x_j)||_2}{||x_i-x_j||}
$$

## **Metric 5: Readability**

$$
R(X) = \frac{1}{d}\Sigma_{i=1}^d|r(X_i, f(X_i))|
$$

$r(x, y)$: Spearman correlation coefficient of two vectors of equal size

## **Metric 6: Clusterability**

$$
Cl(X) = \frac{2}{d * (d-1)}\Sigma_{i,j \in [1,...,d]; i \not ={j}}S(K(f(X_i), f(Xj)))
$$

$K$: clustering
$S$: Silhouette score