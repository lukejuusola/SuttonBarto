### Chapter 2
Select Problems 
- p30. 2.3
    - With probability 1, both greedy parameters will be able to greedily choose the optimal arm. Once the optimal arm is known, an $\epsilon$-greedy policy will choose the best action with probability $p = ( 1 - \epsilon ) + \epsilon / k$. For the case of $\epsilon=.1$ vs $\epsilon = .01$ and $k=10$, the first will pick the optimal action with probability $.9 + .1/10 = .91$ while the second will pick the optimal action with probability $.99 + .01/10 = .991$. With the prior that each arm's mean value is chosen i.i.d from $N(0,1)$, we can assert that the mean reward for a suboptimal action is $0$, while the reward for the optimal action is $R_B^k = \mathbb E_\pi[\max \{N(0,1) \}_1^k]$. By the law of large numbers, we can say that $R_\epsilon^k$ will converge (with probability 1) to $(1 - \epsilon + \epsilon / k)*R_B^k$ and therefore, the $\epsilon=.01$ will outperform the $\epsilon=.1$ policy by $(.991-.91)R_B = .081 \cdot R_B^10$.
- p33. 2.4
- p36. 2.8
    - By definition, UCB explores for the first 10 steps. On the 11th step, the UCB confidence summand is equal across the actions, and therefore, UBC will coincide with the once-sampled greedy choice. This is most probably the optimal action. On the subsequent steps, barring the optimal action being far superior to the suboptimal actions, UCB will return to exploring, and opt for picking suboptimal actions until each action has been sampled sufficiently many times for the mean value difference to overwhelm the confidence summand. 
- [p44. 2.11. Nonstationary Parameter Study](chapters/chapter2/ParameterStudy_NonStationary.ipynb)

Summary and Conclusions
The K-Armed Bandit Problem is the stateless reinforcment learning problem and Sutton and Barto use it here to introduce three key concepts in RL
- Exploration vs Exploitation
    - The question at hand: At any point in time, we are given the option: should we sample the state space or should we try to collect reward? This is a problem foundational in human life and any problem that requires the real time acquisition of experience. Modelling the state space is valuable but comes at an opportunity cost. The question of how to balance these is the Exploration vs Exploitation Trade off. The simplest method is $epsilon$-greedy, for the bandit problem the Upper-Confidence-Bound algorithm clearly outperforms. We'll explore what to do in more complex situations later in the book, but *sampling while playing* is fundamental in the reinforcement learning paradigm. Information is valuable but marginalizing( in a stationary problem ), at some point, we should be satisifed with what we know and cash it in. How and how much to explore is the name of the game.
- Nonstationary Problems
    - While traditional ML focuses solely on the case of i.i.d. stationary distributions to estimate, Reinforcement Learning problems are very often *non-stationary.* That is, the problem to learn can change under your feet. A picture of a cat will always be a picture of a cat*. The rewards that a slot machine gives may be quickly changing. Building agents robust to non-stationarity or acutely aware of it will be one of the primary goals.
- Iterative Policy Gradient Updates
    - Many simple algorithms have streaming versions, such as the sample mean. Fast computation and low memory utilization are going to be paramount to making these systems scalable.
    - You can define a stochastic (softmax) policy where weights are given by preferences rather than reward estimates. Using an modern SGD, we can optimize this directly, and I suspect this will come in handy once we're considering DeepRL. 

### Chapter 3
Exercises
- p56. 3.6
- p56. 3.10
    - $G = \sum_0^\infty \gamma^k$
    - $1 + \gamma G = \gamma^0 + \sum_1^\infty \gamma^k = G$
    - $G = \frac{1}{1 - \gamma}$
- p58. 3.11, 3.12, 3.13
    - 3.11
        - $\mathbb E_\pi[ R | S_t=s, A_t=a ] = \sum_{s'} \sum_r p(s', r | s, a) * r$
    - 3.12
        - $v_\pi(s) = \mathbb E_\pi[G_t | S_t = s] = \sum_{a \in A} \pi(a|s)q_\pi(s, a)$
    - 3.13
        - $q_\pi(s, a) = \mathbb E_pi[G_t|S_t=s,A_t=a] = \mathbb E_pi[R_t + \gamma G_{t + 1}|S_t=s,A_t=a]$
        - $= \sum_{s'} \sum_r p(s', r | s, a ) ( r + \gamma \mathbb \mathbb E_\pi_\pi [G_{t+1} | S_{t+1} = s'] )$
        - $= \sum_{s'} \sum_r p(s', r | s, a ) ( r + \gamma v_\pi( s' ) )$
- p61. 3.15, 3.16
- p61. 3.17 Bellman Equation for $q_\pi$:
    - $q_\pi(s, a) = \mathbb E_\pi[ G_t | S_t=s,A_t=a ] = \mathbb E_\pi[ R_t + \gamma G_{t+1} | S_t=s,A_t=a ]$
    - $= \sum_{s'} \sum_r p(s', r | s, a) ( r + \gamma v_pi(s') )$
    - $= \sum_{s'} \sum_r p(s', r | s, a) ( r + \gamma \sum_{a'} \pi(s', a') q_\pi(s', a') )$
- p62. 3.18
    - $v_\pi(s) = \mathbb E_\pi[ G_t | S_t=s ] = \sum_a \pi(a|s) \mathbb E_\pi[ G_t | S_t=s,A_t=a ]$
    - $v_\pi(s) = \mathbb E_\pi[ G_t | S_t=s ] = \sum_a \pi(a|s) q_\pi(s, a)$
- p66. 3.22
    - $\pi = left$
        - $\gamma = 0.0: v = \sum_0^\infty (1 \gamma^{2k}) + ( 0 \gamma^{2k+1})) = 1$
        - $\gamma = 0.5: v = \sum_0^\infty (1 \gamma^{2k}) + ( 0 \gamma^{2k+1})) = \sum_0^\infty (\gamma^2)^k = \frac{1}{1 - \gamma^2} = \frac{1}{1 - .75} = 4$
        - $\gamma = 0.9: v = \sum_0^\infty (1 \gamma^{2k}) + ( 0 \gamma^{2k+1})) = \sum_0^\infty (\gamma^2)^k = \frac{1}{1 - \gamma^2} = \frac{1}{1 - .81} = 5.26$
    - $\pi = right$
        - $\gamma = 0.0: v = \sum_0^\infty (0 \gamma^{2k}) + ( 2 \gamma^{2k+1})) = 0$
        - $\gamma = 0.5: v = \sum_0^\infty (0 \gamma^{2k}) + ( 2 \gamma^{2k+1})) = 2 \gamma \sum_0^\infty (\gamma^2)^k = \frac{2 \gamma}{1 - \gamma^2} = \frac{1}{1 - .75} = 8 \gamma = 4$
        - $\gamma = 0.9: v = \sum_0^\infty (0 \gamma^{2k}) + ( 2 \gamma^{2k+1})) = 2 \gamma \sum_0^\infty (\gamma^2)^k = \frac{2 \gamma}{1 - \gamma^2} = \frac{1.8}{1 - .81} = 9.47$
- p67. 3.25,3.26,3.27,3.28,3.29 

Select Problems
-