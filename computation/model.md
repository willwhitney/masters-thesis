## Controller-function networks

![The controller and layers of the system. The controller provides weights on each layer as a function of the data. This shows three layers, but there can be many more.](controller_network_small.png){#fig:controller_network}

The proposed model, the controller-function network (CFN) generates an output for a particular timestep via the following steps (shown in [@Fig:controller_network]):

1. The input tensor is fed into the controller
1. The controller decides which layers are most appropriate for processing this input
1. The controller outputs a weighting vector reflecting how much output it wants from each of the layers
1. The input tensor is fed into each layer (in parallel)
1. The outputs from each layer are multiplied by their respective weights from the controller
1. The weighted outputs from all the layers are summed together and output. This is the output of the whole network for this timestep.

Essentially the idea is that at each timestep, the controller examines the input that it gets, then up- or down-regulates the activities of the various "functions" (single-layer NNs) to best deal with this input. Since the controller is an LSTM, it can store information about the inputs it has received before, meaning that in a time series or language setting it can make weighting decisions contextually.

As this model is differentiable throughout, it can be trained with the standard backpropagation through time (BPTT) algorithm for stochastic gradient descent.

By setting weights over each of the layers in the network, the controller scales not only the output of each layer, but also the error gradient that it receives. This means that in a given timestep, the layers which have very low weights on their output will be nearly unchanged by the learning process. That is, functions which are not used are not forgotten.

In an ordinary feedforward neural network, the only way for the network to prevent learning in a particular node is for it to learn connection strengths very near zero for that node. This takes many training examples, and functionally removes that node from the computation graph.

This system, by comparison, can decide that a set of nodes is or is not relevant on an input-by-input basis.


### Relationship to mixture of experts

This architecture is closely related to the mixture of experts model proposed by Jacobs et al. [-@jacobs1991task], in which several different task-specific "expert" networks each contribute in linear combination to the output of the overall network.

However, this model has two key differences from the mixture of experts:

1. **The gating network is an LSTM.** This means that the gating network (or controller, in my terminology) can easily learn fixed sequential procedures for certain types of input. This allow the model to be iterated for several steps, composing its operations into more complex ones. See [@Sec:multistep].
1. **The training results in decoupled functions.** I employ a novel continuation method for training the CFN that allows for easy training, yet results in a final representation which has 


### Hard and soft decisions

Training neural models which make "hard" decisions can be quite challenging; in the general case, such models must be trained by REINFORCE-like gradient estimation methods [@williams1992simple]. Yet under many circumstances, such hard decisions are necessary for computational considerations; in fully-differentiable models such as the NTM [@graves2014neural] or end-to-end memory networks [@sukhbaatar2015end], the computational complexity of a single evaluation increases linearly with the size of the memory. These "soft" decisions involve considering every possible option.

In more complex computational tasks, such as those faced by [@reed2015neural], there may be a large number of steps before any result is produced by the model, and each step can require a discrete action (move the pointer either left or right; move the model either up or down). Such models naïvely have branching which is exponential of the form $O(k^t)$, where $k$ is the number of options at each timestep, and $t$ is the number of timesteps before producing an output. Using a REINFORCE algorithm to estimate the true gradient is possible, but slow and unreliable [@zaremba2015reinforcement]. This branching factor is what led [@reed2015neural] to adopt their strongly-supervised training technique.

A straightforward (if inelegant) solution is to composite the outcome from all of these branches at the end of each timestep. For example, a pointer could be modeled as interpolating between two memory cells instead of having a discrete location. Then when the controller produces the distribution of actions "left 0.7, right 0.3", the model can move the pointer left by 0.7 instead of sampling from $Bernoulli(0.7)$.

While such techniques, make the learning process tractable when available, they result in much more highly entangled representations (e.g. reading from a every location in a memory at once). Furthermore, they must always incur a complexity cost linear in the number of options, just as the memory models have cost linear in the number of options of memory locations to read from.

This solution is not always available. In classic reinforcement learning tasks, the agent will only be in a situation once, and it cannot 70% fly to Germany or 20% accept a PhD position.

The CFN exists in the space of models for which this soft-decision solution is available. While in the ideal case we would like to select exactly one function to use at each timestep, this problem is quite difficult to optimize, for early in training the functions are not yet differentiated. By contrast, the soft-decision version which uses a weighted sum of the outputs of each function learns quite quickly. However, the solutions produced by this weighted sum training are highly entangled and always involve a linear combination of all the functions, with no clear differentiation.

From scratch, we can either train a system that works, or a system that has good representations. What we need is a way to go from a working solution to a good solution.

### Continuation methods

Continuation methods are a widely-used technique for approaching difficult optimization problems.

> In optimization by continuation, a
transformation of the nonconvex function to an easy-to-minimize
function is considered. The method then progressively
converts the easy problem back to the original
function, while following the path of the minimizer. [@mobahi2015theoretical]

As described in [@mobahi2015theoretical], continuations include ideas as ubiquitous as curriculum learning or deterministic annealing, and that paper provides an extensive list of examples. In the quest for good solutions to hard-decision problems, continuation methods are a natural tool.

### Training with noisy decisions

In order to construct a continuation between soft and hard decisions, the CFN combines two tools: weight sharpening and noise.

Weight sharpening is a technique used by [@graves2014neural], which works by taking a distribution vector of weights $w \in [0,1]^n$, and a sharpening parameter $\gamma \ge 1$ and transforming $w$ as follows:

$$w_i' = \frac{w_i^{\gamma}}{\sum_j w_j^{\gamma}}$$

By taking this $[0,1]^n$ vector to an exponent, sharpening increases the relative differences between the weights in $w$. Renormalizing makes $w$ a distribution once again, but now it has been stretched; large values are larger, i.e. the modes have higher probability. In the CFN, I take one further step: adding noise.

$$w_i' = \frac{\big(w_i + \mathcal{N}(0, \sigma^2)\big)^{\gamma}}{\sum_j w_j^{\gamma}}$$

During the training of the CFN, sharpening is applied to the vector of weights produced by the controller, and the sharpening parameter $\gamma$ is gradually increased on a schedule. By itself, this would not transform the outputs of the controller, as it can simply learn the inverse function to continue to produce the same output. However, the addition of noise before sharpening makes similar weights highly unstable: 

$$\frac{[0.49, 0.51]^{100}}{(0.49^{100} + 0.51^{100})} = [0.018, 0.982]$$

At the end of training, this forces the CFN to either make a hard decision or face massive uncertainty in its output. By slowly increasing the sharpening parameter on a schedule, the controller can gradually learn to make harder and harder decisions. In practice this method works very well, resulting in perfectly binary decisions at the end of training and correct factorization of the primitives, each into its own function layer.

