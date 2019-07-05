{% include head.html %}

# EM Algorithm and Variational Bayesian Inference

Here I provide a concise description of the general idea of the methods.

In short, both methods serve to estimate parameters in a statistical model with partially observed samples
taken from a distribution.

## Maximum Log-Likelihood Estimator

Let's begin with something simple. Here I explain well known method in a not so well known way.

Suppose we have a bunch of samples $\{x_1, x_2, ..., x_N\}$ from an arbitrary probability space $(X, \Sigma, P)$.
Well, not exactly arbitrary. For further needs we demand $X$ to be *separable metric space* with
$\Sigma$ being *Borel algebra*.
Then, we also postulate the existance of some reference measure on $(X, \Sigma)$. Denote it $\mu$. This measure
should meet *Lebesgue differentiation theorem* requirments for at least sample points $x_k$. Namely for
each complex-valued measurable function $f$ the following should be true:

$$
\lim_{\epsilon \to 0}{\frac{1}{\mu(B_{x_k, \epsilon})} \int_{B_{x_k, \epsilon}}{f(x)d\mu(x)}} = f(x_k)
$$

Where $B_{x_k, \epsilon}$ is an open ball with center at $x_k$ and radius $\epsilon$. For example, $X$ can simply
be $\mathbf{R}^n$ and $\mu$ can be standard Lebesgue measure.

The next step is to define a family of candidate distributions for $P$. This is a *statistical model*.
It can be something simple like family of Gaussian distributions or something complicated like
family of neural networks with stochastic input. The only requirement is *absolute continuity with respect
to $\mu$* each of these distributions. In other words, using Radon-Nikodym theorem, we can define a
pdf $p(x)$ for each distribution $P(A)$ from that family:

$$
P(A) = \int_{A}{p(x)d\mu(x)}
$$

Typically, these models are parametrized with a real-valued vector of parameters $\mathbf{\theta}$.
Our goal is to find an algorithm that estimates $\mathbf{\theta}$ based on given data.

We have a family of distributions $P^{\theta}(A)$. Our goal is to choose "the best" one.
Firstly, let's consider a wider family of probability distributions.
Namely, all possible probability measures over $(X, \Sigma)$.
The best distribution which fits our data is the average of Dirac measures $\delta_{x_k}(A)$
centered at sample point. This distribution is called *empirical distribution* of data or *observed point*:

$$
Q(A) = \frac{1}{N} \sum_{k=1}^N{\delta_{x_k}(A)}
$$

Then, simply find the distribution $P^{\theta^*}(A)$ in our statistical model which minimizes KL-divergence
from $Q(x)$ (Later I will write a post about different divergences as measures of dissimilarity between
distributions and why we typically choose KL-divergence).

$$
D_{KL}(Q || P^{\theta}) = \int_{X}{\log{\frac{dQ}{dP^{\theta}}} dQ}
$$

However, in order to use KL-divergence, we need to guarantee existence of Radon-Nikodym derivative
$\frac{dQ}{dP^{\theta}}$.
Unfortunately it doesn't exist in our case.
One solution is to change our idealized distribution $Q(A)$ to something
more plausible. Introduce a small $\epsilon$-ball centered at each
sample point, name it $B_{x_k, \epsilon}$. $\epsilon$ should be small enough to not allow intersection of
these balls. It is possible since $X$ is a metric space hence *Hausdorff*.
Finally, we define our *corrected empirical distribution* as a mixture of uniform distributions on these balls.

$$
\widetilde{Q}(A) = \frac{1}{N}\sum_{k=1}^N{\widetilde{\delta}_{x_k}(A)}
$$

Where

$$
\widetilde{\delta}_{x_k}(A) = \frac{1}{\mu(B_{x_k, \epsilon})} \mu(A \cap B_{x_k, \epsilon})
$$

Also worth to notice that $\widetilde{Q}$ is absolute continuous with respect to $\mu$. And
by Radon-Nicodym theorem could be written as

$$
\widetilde{Q}(A) = \int_{A}{\widetilde{q}(x)d\mu(x)}
$$

With

$$
\widetilde{q}(x) = \left\{
                \begin{array}{ll}
                  \frac{1}{N\mu(B_{x_k, \epsilon})}\text{,  }x \in B_{x_k, \epsilon}\text{ for some }k,\\
                  0\text{, otherwise}
                \end{array}
              \right.
$$

So $\widetilde{q}$ is a pdf for $\widetilde{Q}$.
Finally, we can define Radon-Nikodym derivative $\frac{d\widetilde{Q}}{dP^{\theta}}$:

$$
\frac{d\widetilde{Q}}{dP^{\theta}} = \frac{\widetilde{q}(x)}{p_{\theta}(x)}
$$

Now we can write KL-divergence $D_{KL}(\widetilde{Q} \vert \vert P^{\theta})$

$$
\begin{aligned}
D_{KL}(\widetilde{Q} \vert \vert P^{\theta}) &= \int_{X}{\log{\frac{d\widetilde{Q}(x)}{dP^{\theta}(x)}} d\widetilde{Q}(x)} \\
&= \frac{1}{N\mu(B_{x_k, \epsilon})}\sum_{k=1}^N{\int_{B_{x_k, \epsilon}}{\log{\frac{1}{Np_{\theta}(x)\mu(B_{x_k, \epsilon})}}d\mu(x)}} \\
&= -\frac{1}{N}\sum_{k=1}^N{\frac{1}{\mu(B_{x_k, \epsilon})}\int_{B_{x_k, \epsilon}}{\log{p_{\theta}(x)}d\mu(x)}} + Const
\end{aligned}
$$

Notice that our initial assumptions give us

$$
\frac{1}{\mu(B_{x_k, \epsilon})}\int_{B_{x_k, \epsilon}}{\log{p_{\theta}(x)}d\mu(x)} \approx \log{p_{\theta}(x_k)}
$$

So, finally

$$
D_{KL}(\widetilde{Q} \vert \vert P^{\theta}) \approx -\frac{1}{N} \sum_{k=1}^N{\log{p_{\theta}(x_k)}} + Const
$$

![](kl_no_hidden.svg?raw=true)

Conclusion: minimization of KL-divergence of desired distribution from the corrected empirical distribution
is equivalent to maximization of log-likelihood.

Another interesting observation is disregarding the choice of reference measure $\mu$ the resulting
expression is the same. Yes, pdf $p_{\theta}$ will be different for different $\mu$, but difference is in
the constant.


## EM Algorithm

Suppose each sample $x_k$ is coupled with another peace of information which is hidden from us, call it $z_k$.
$z_k$ comes from another measurable space $(Z, \Sigma_z)$ equiped with its own reference measure $\mu_Z$
(reference measure on $X$ is renamed to $\mu_X$).
Couples $(x_k, z_k)$ are samples from some unknown distribution $P$ defined on
measurable space $(X \times Z, \Sigma_x \otimes \Sigma_z)$. We also define new reference measure
on this space as product measure $\mu = \mu_X \times \mu_Z$.

Define a statistical model (family of distributions) $P^{\theta}(A)$ acting on
$(X \times Z, \Sigma_x \otimes \Sigma_z)$. Our goal is, again, pick "the best"
distribution from this model based only on $x_k$ part of samples.

We want to define a corrected empirical distribution $\widetilde{Q}(A)$ in the similar way to the previous section.
Decompose $\widetilde{Q}(A)$ using marginal distribution %\widetilde{Q}_X(E_x) = \widetilde{Q} \circ \pi_X^{-1}%
defined on $(X, \Sigma_x)$ and family of conditional distributions \(\widetilde{Q}_{Z \vert x}(E_z)\)
parametrized by points from $X$ and defined on $(Z, \Sigma_z)$.
All these probability measures are absolute continuous with respect to their
corresponding reference measures and have pdfs.

$$
\widetilde{Q}_X(A_x) = \int_{A_x}{\widetilde{q}_X(x)d\mu_X(x)}
$$

$$
\widetilde{Q}_{Z \vert x}(A_z) = \int_{A_z}{\widetilde{q}_Z(z \vert x) d\mu_Z(z)}
$$

Resulting decomposition:

$$
\begin{aligned}
\widetilde{Q}(A) &= \int_{X}{\widetilde{Q}_{Z \vert x}(A^x)d\widetilde{Q}_X(x)}\\
&= \int_{X}\int_{A^x}{\widetilde{q}_X(x)\widetilde{q}_Z(z \vert x)d\mu_Z(z)d\mu_X(x)} \\
&= \int_{A}{\widetilde{q}_X(x)\widetilde{q}_Z(z \vert x) d\mu(x, z)}
\end{aligned}
$$

Where $A^x = \{z \in Z \vert (x, z) \in A\}$

Using results from the previous section we put

$$
\widetilde{Q}_X(A_x) = \frac{1}{N}\sum_{k=1}^N{\widetilde{\delta}_{x_k}(A_x)}
$$

Using that

$$
\begin{aligned}
\widetilde{Q}(A) &= \int_{X}{\widetilde{Q}_{Z \vert x}(A^x)d\widetilde{Q}_X(x)} \\
&= \frac{1}{N}\sum_{k=1}^N{\frac{1}{\mu_X(B_{x_k, \epsilon})} \int_{B_{x_k, \epsilon}}{\widetilde{Q}_{Z \vert x}(A^x)d\mu_X(x)}} \\
&\approx \frac{1}{N}\sum_{k=1}^N{\widetilde{Q}_{Z \vert x_k}(A^{x_k})}
\end{aligned}
$$

But we have no information about what $\widetilde{Q}_{Z \vert x_k}$ could possibly be. So instead of one
corrected empirical distribution we have a whole family of them. And the problem now is to minimize
KL-divergence from this family.

![](kl_hidden.svg?raw=true)

We also assume that distributions from our statistical model are absolute continuous with respect to $\mu$:

$$
P^{\theta}(A) = \int_{A}{p_{\theta}(x, z)d\mu(x,z)} = \int_{A}{p_{\theta}(x)p_{\theta}(z \vert x)d\mu(x, z)}
$$

Then KL Divergence is

$$
\begin{aligned}
D_{KL}(\widetilde{Q} \vert \vert P^{\theta}) &= \int_{X \times Z}{\log{\frac{d\widetilde{Q}}{dP^{\theta}}(x, z)}d\widetilde{Q}(x, z)} \\
&= \int_{X}\int_{Z}{\log{\frac{\widetilde{q}_X(x)\widetilde{q}_Z(z \vert x)}{p_{\theta}(x, z)}}\widetilde{q}_Z(z \vert x)d\mu_Z(z)d\widetilde{Q}_X(x)} \\
&= \frac{1}{N}\sum_{k=1}^N{\frac{1}{\mu_X(B_{x_k, \epsilon})} \int_{B_{x_k, \epsilon}}\int_{Z}{\log{\frac{\widetilde{q}_Z(z \vert x)}{N \mu_X(B_{x_k, \epsilon)} p_{\theta}(x, z)}}\widetilde{q}_Z(z \vert x)d\mu_Z(z)d\mu_X(x)}} \\
&\approx \frac{1}{N}\sum_{k=1}^N{\int_{Z}{\log{\frac{\widetilde{q}_Z(z \vert x_k)}{N\mu_X(B_{x_k, \epsilon})p_{\theta}(x_k, z)}}\widetilde{q}_Z(z \vert x_k) d\mu_Z(z)}} \\
&= \frac{1}{N}\sum_{k=1}^N{\int_{Z}{\log{\frac{\widetilde{q}(z \vert x_k)}{p_{\theta}(x_k, z)}}\widetilde{q}_Z(z \vert x_k)d\mu_Z(z)}} + Const
\end{aligned}
$$

The goal is to find $\theta$ as well as $\widetilde{q}_Z(z \vert x_k)$ for each $k$ which minimize the latter
expression.

$$
\begin{aligned}
D_{KL}(\widetilde{Q} \vert \vert P^{\theta}) &= \frac{1}{N}\sum_{k=1}^N{\int_{Z}{\log{\frac{\widetilde{q}(z \vert x_k)}{p_{\theta}(x_k, z)}}\widetilde{q}_Z(z \vert x_k)d\mu_Z(z)}} + Const \\
&= \frac{1}{N}\sum_{k=1}^N{\int_{Z}{\log{\frac{\widetilde{q}(z \vert x_k)}{p_{\theta}(x_k)p_{\theta}(z \vert x_k)}}\widetilde{q}_Z(z \vert x_k)d\mu_Z(z)}} + Const \\
&= \frac{1}{N}\sum_{k=1}^N{D_{KL}(\widetilde{Q}_{Z \vert x_k} \vert \vert P_{Z \vert x_k}^{\theta}) - \int_{Z}{\log{p_{\theta}(x_k)}\widetilde{q}_Z(z \vert x_k)d\mu_Z(z)}} + Const \\
&= \frac{1}{N}\sum_{k=1}^N{D_{KL}(\widetilde{Q}_{Z \vert x_k} \vert \vert P_{Z \vert x_k}^{\theta}) - \log{p_{\theta}(x_k)}} + Const
\end{aligned}
$$

From here we can see that $\widetilde{q}_Z(z \vert x_k) = p_{\theta}(z \vert x_k)$ minimizes this expression
and further minimization with respect to $\theta$ leads to the same solution as
maximization of marginal log likelihood of desired distribution. However direct maximization of
$\sum_{k=1}^{N}{\log{p_{\theta}(x_k)}}$ may not be tractable. We can expand expression for
$D_{KL}(\widetilde{Q} \vert \vert P^{\theta})$ in yet another way

$$
\begin{aligned}
D_{KL}(\widetilde{Q} \vert \vert P^{\theta}) &= \frac{1}{N}\sum_{k=1}^N{\int_{Z}{\log{\frac{\widetilde{q}(z \vert x_k)}{p_{\theta}(x_k, z)}}\widetilde{q}_Z(z \vert x_k)d\mu_Z(z)}} + Const \\
&= \frac{1}{N}\sum{-\mathbb{E}_{\widetilde{q}_Z(z \vert x_k)}[\log{p_{\theta}(x_k, z)}] + H(\widetilde{Q}_{Z \vert x_k})} + Const
\end{aligned}
$$

If $\widetilde{q}_Z(z \vert x_k)$ stays fixed, we can maximize
$\mathbb{E}_{\widetilde{q}_Z(z \vert x_k)}[\log{p_{\theta}(x_k, z)}]$ either analytically or by Monte-Carlo
sampling from $\widetilde{q}_Z(z \vert x_k)$ and solving regular inference problem from samples $(x_k, z_k)$.

Finaly we arrive to the algorithm itself:

1. Initialize $\theta_0$, $t = 0$.
2. Fix $\widetilde{q}_Z(z \vert x_k) = p_{\theta_t}(z \vert x_k)$.
3. Solve $\theta_{t+1} = \mathop{\text{argmax}}_{\theta}{\mathbb{E}_{\widetilde{q}_Z(z \vert x_k)}[\log{p_{\theta}(x_k, z)}]}$.
4. If $\vert \theta_{t+1} - \theta_{t} \vert < \epsilon$ then stop, else go to step 2.

Notice, for each step we never increase the value of $D_{KL}(\widetilde{Q} \vert \vert P^{\theta})$.
Therefore algorithm converge.
