## Model

![**Model Architecture.** Deep Convolutional Inverse Graphics Network (DC-IGN) has an encoder and a decoder. We follow the variational autoencoder [@kingma2013auto] architecture with variations. The encoder consists of several layers of convolutions followed by max-pooling and the decoder has several layers of unpooling (upsampling using nearest neighbors) followed by convolution. (a) During training, data $x$ is passed through the encoder to produce the posterior approximation $Q(z_i|x)$, where $z_i$ consists of scene latent variables such as pose, light, texture or shape. In order to learn parameters in DC-IGN, gradients are back-propagated using stochastic gradient descent using the following variational object function: $-log(P(x|z_i)) + KL(Q(z_i|x)||P(z_i))$ for every $z_i$. We can force DC-IGN to learn a disentangled representation by showing mini-batches with a set of inactive and active transformations (e.g. face rotating, light sweeping in some direction etc). (b) During test, data $x$ can be passed through the encoder to get latents $z_i$. Images can be re-rendered to different viewpoints, lighting conditions, shape variations, etc by setting the appropriate graphics code group $(z_i)$, which is how one would manipulate an off-the-shelf 3D graphics engine.](../figures/overview.pdf){#fig:overview}

As shown in Figure [@fig:overview], the basic structure of the Deep Convolutional Inverse Graphics Network (DC-IGN) consists of two parts: an encoder network which captures a distribution over graphics codes $Z$ given data $x$ and a decoder network which learns a conditional distribution to produce an approximation $\hat{x}$ given $Z$. $Z$ can be a disentangled representation containing a factored set of latent variables $z_i \in Z$ such as pose, light and shape. This is important in learning a meaningful approximation of a 3D graphics engine and helps tease apart the generalization capability of the model with respect to different types of transformations.

![**Structure of the representation vector.** $\phi$ is the azimuth of the face, $\alpha$ is the elevation of the face with respect to the camera, and $\phi_L$ is the azimuth of the light source.](../figures/latents_legend.pdf){#fig:latentslegend width=70%}

Let us denote the encoder output of DC-IGN to be $y_e = encoder(x)$. The encoder output is used to parametrize the variational approximation $Q(z_i|y_e)$, where $Q$ is chosen to be a multivariate normal distribution. There are two reasons for using this parametrization:

1. Gradients of samples with respect to parameters $\theta$ of $Q$ can be easily obtained using the reparametrization trick proposed in [@kingma2013auto]
2. Various statistical shape models trained on 3D scanner data such as faces have the same multivariate normal latent distribution [@paysan2009face].

Given that model parameters $W_e$ connect $y_e$ and $z_i$, the distribution parameters $\theta = (\mu_{z_i}, \Sigma_{z_i})$ and latents $Z$ can then be expressed as:

$$\mu_{z} = W_e  y_e$$
$$\Sigma_{z} &= \text{diag}(\exp(W_e  y_e))$$
$$\forall{i}, z_i &\sim \mathcal{N}(\mu_{z_i}, \Sigma_{z_i})$$

We present a novel training procedure which allows networks to be trained to have disentangled and interpretable representations.

### Training with Specific Transformations {#sec:specifictransforms}

![**Training on a minibatch in which only $\phi$, the azimuth angle of the face, changes.** During the forward step, the output from each component $z_i \neq z_1$ of the encoder is altered to be the same for each sample in the batch. This reflects the fact that the generating variables of the image (e.g. the identity of the face) which correspond to the desired values of these latents are unchanged throughout the batch. By holding these outputs constant throughout the batch, the single neuron $z_1$ is forced to explain all the variance within the batch, i.e. the full range of changes to the image caused by changing $\phi$. During the backward step $z_1$ is the only neuron which receives a gradient signal from the attempted reconstruction, and all $z_i \neq z_1$ receive a signal which nudges them to be closer to their respective averages over the batch. During the complete training process, after this batch, another batch is selected at random; it likewise contains variations of only one of ${\phi, \alpha, \phi_L, intrinsic}$; all neurons which do not correspond to the selected latent are clamped; and the training proceeds.](../figures/remastered_training_diagram.pdf){#fig:selectivetraining}

The main goal of this work is to learn a representation of the data which consists of disentangled and semantically interpretable latent variables. We would like only a small subset of the latent variables to change for sequences of inputs corresponding to real-world events.

One natural choice of target representation for information about scenes is that already designed for use in graphics engines. If we can deconstruct a face image by splitting it into variables for pose, light, and shape, we can trivially represent the same transformations that these variables are used for in graphics applications. [@Fig:latentslegend] depicts the representation which we will attempt to learn.

With this goal in mind, we perform a training procedure which directly targets this definition of disentanglement. We organize our data into mini-batches corresponding to changes in only a single scene variable (azimuth angle, elevation angle, azimuth angle of the light source); these are transformations which might occur in the real world. We will term these the _extrinsic_ variables, and they are represented by the components $z_{1,2,3}$ of the encoding.

We also generate mini-batches in which the three extrinsic scene variables are held fixed but all other properties of the face change. That is, these batches consist of many different faces under the same viewing conditions and pose. These _intrinsic_ properties of the model, which describe identity, shape, expression, etc., are represented by the remainder of the latent variables $z_{[4,200]}$. These mini-batches varying intrinsic properties are interspersed stochastically with those varying the extrinsic properties.

We train this representation using SGVB, but we make some key adjustments to the outputs of the encoder and the gradients which train it. The procedure ([@Fig:selectivetraining]) is as follows.


1. Select at random a latent variable $z_{train}$ which we wish to correspond to one of {azimuth angle, elevation angle, azimuth of light source, intrinsic properties}.
1. Select at random a mini-batch in which that only that variable changes.
1. Show the network each example in the minibatch and capture its latent representation for that example $z^k$.
1. Calculate the average of those representation vectors over the entire batch.
1. Before putting the encoder's output into the decoder, replace the values $z_i \neq z_{train}$ with their averages over the entire batch. These outputs are "clamped".
1. Calculate reconstruction error and backpropagate as per SGVB in the decoder.
1. Replace the gradients for the latents $z_i \neq z_{train}$ (the clamped neurons) with their difference from the mean (see [@Sec:targetedinvar]). The gradient at $z_{train}$ is passed through unchanged.
1. Continue backpropagation through the encoder using the modified gradient.

Since the intrinsic representation is much higher-dimensional than the extrinsic ones, it requires more training. Accordingly we select the type of batch to use in a ratio of about 1:1:1:10, azimuth : elevation : lighting : intrinsic; we arrived at this ratio after extensive testing, and it works well for both of our datasets.

This training procedure works to train both the encoder and decoder to represent certain properties of the data in a specific neuron. By clamping the output of all but one of the neurons, we force the decoder to recreate all the variation in that batch using only the changes in that one neuron's value. By clamping the gradients, we train the encoder to put all the information about the variations in the batch into one output neuron.

### Invariance Targeting {#sec:targetedinvar}
By training with only one transformation at a time, we are encouraging certain neurons to contain specific information; this is equivariance. But we also wish to explicitly _discourage_ them from having _other_ information; that is, we want them to be invariant to other transformations. Since our mini-batches of training data consist of only one transformation per batch, then this goal corresponds to having all but one of the output neurons of the encoder give the same output for every image in the batch.

To encourage this property of the DC-IGN, we train all the neurons which correspond to the inactive transformations with an error gradient equal to their difference from the mean. It is simplest to think about this gradient as acting on the set of subvectors $z_{inactive}$ from the encoder for each input in the batch. Each of these $z_{inactive}$'s will be pointing to a close-together but not identical point in a high-dimensional space; the invariance training signal will push them all closer together. We don't care where they are; the network can represent the face shown in this batch however it likes. We only care that the network always represents it as still being the same face, no matter which way it's facing. This regularizing force needs to be scaled to be much smaller than the true training signal, otherwise it can overwhelm the reconstruction goal. Empirically, a factor of $1/100$ works well.
