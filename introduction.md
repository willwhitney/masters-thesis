# Introduction

Representation is one of the most fundamental problems in machine learning. It underlies such varied fields as vision, speech recognition, natural language processing, reinforcement learning, and graphics. Yet the question of what makes a good representation is a deceptively complex one. On the one hand, we would like representations which perform well on real-world tasks. On the other, we would like to be able to interpret these representations, and they should be useful for tasks beyond those explicit in their initial design.

Presently representations come in two varieties: those that are designed, and those that are learned from data. Designed representations can perfectly match our desire for structured reuse and interpretability, while learned representations require no expert knowledge yet outperform designed features on practically every task that has sufficient data.

This tension has been the source of great debate in the community. Clearly a representation which has some factorizable structure can be more readily reused in part or in whole for some new task. Much more than being just an issue of interpretation, this concern has a very practical focus on generalization; it is unreasonable to spend a tremendous amount of data building a new representation for every single task, even when those tasks have strong commonalities. Since we have knowledge about the true structure of the problem, we can design a representation which is factorized and thus reusable.

Learned representations take a very different approach to the problem. Instead of attempting to incorporate expert knowledge of a domain to create a representation which will be broadly useful, a learned representation is simply the solution to a single optimization problem. It is custom-designed to solve precisely the task it was trained on, and while it may be possible to reverse-engineer such a representation for reuse elsewhere, it is typically unclear how to do so and how useful it will be in a secondary setting.

Despite the obvious advantages of structured representations, those we design are inherently limited by our understanding of the problem. Perhaps it is possible to design image features with spokes and circles that will be able to distinguish a bike wheel from a car wheel, but there are a million subtle clues that no human would think to include. As a result, in domain after domain, as the datasets have grown larger, representations learned by deep neural networks have come to dominate.

The dominance of optimization-based representation learning is unavoidable and in fact hugely beneficial to the field. However, the weaknesses of these learned representations is not inherent in their nature; it merely reflects the limits of our current tasks and techniques.

This thesis represents an effort to bring together the advantages of each of these techniques to learn representations which perform well, yet have valuable structure. Using the two domains of graphics and programs, I will discuss the rationale, techniques, and results of bringing structure to neural representations.

## Document overview

The next chapter discusses various criteria for assessing the quality of a representation.

In the following two chapters, I use these criteria to discuss representations in the domains of graphics and computer programs. Each chapter begins with an overview of the problems in the field and related work, then moves on to a description of the specific problem I address, my methods, and the results.

In the final chapter I discuss the significance of this work to the field, future research directions, and the philosophy of model design in the deep learning era.


# Desiderata for representations

When evaluating a representation, it is valuable to have a clear set of goals. Several of the goals stated here have substantial overlap, and to some degree a representation which perfectly addresses one may automatically fulfill another as well. However, each of them provides a distinct benefit, and their significance must be considered with respect to those benefits.

## Disentangled

A representation which is _disentangled_ for a particular dataset is one which is sparse over the transformations present in that data [@bengio2013representation]. For example, given a dataset of indoor videos, a representation that explicitly represents whether or not the lights are on is more disentangled than a representation composed of raw pixels. This is because for the common transformation of flipping the light switch, the first representation will only change in only that single dimension (light on or off), whereas the second will change in every single dimension (pixel).

For a representation to be disentangled implies that it factorizes some latent cause or causes of variation. If there are two causes for the transformations in the data which do not always happen together and which are distinguishable, a maximally disentangled representation will have a structure that separates those causes. In the indoor scenes example above, there might be two sources of lighting: sunlight and electric lights. Since transformations in each of these latent variables occur independently, it is more sparse to represent them separately.

The most disentangled possible representation for some data is a graphical model expressing the "true" generative process for that data. In graphics this model might represent each object in a room, with its pose in the scene and its intrinsic reflectance characteristics, and the sources of lighting. For real-world transformations involving motion, only the pose of each object needs to be updated. As the lighting shifts, nothing about the representation of the objects needs to be changed; the visual appearance of the object can be recalculated from the new lighting variables.

In a certain light, all of science is one big unsupervised learning problem in which we search for the most disentangled representation of the world around us.

## Interpretable

An _interpretable_ representation is, simply enough, one that is easy for humans to understand. A human should be able to make predictions about what changes in the source domain would do in the representation domain and vice versa. In a graphics engine's representation of a scene, for example, it is easy for a person to predict things like "What would the image (source domain) look like if I changed the angle of this chair (representation domain) by 90°?" By contrast, in the representation of a classically-trained autoencoder, it is practically impossible for a person to visualize the image that would be generated if some particular component were changed.

Interpretability is closely related with disentanglement. This is because, in "human" domains of data like vision and audition, humans are remarkably good at inferring generative structure, and tend to internally use highly disentangled representations. However, this relationship only holds for datasets which are similar to human experience. One could construct a dataset of videos in which the most common transformation between frames was for each pixel in the image to change its hue by an amount proportional to the number of characters in the Arabic name of the object shown in that pixel. The most disentangled representation of these videos would perfectly match this structure, but this disentangled representation would be less interpretable than a table of English names of objects and how much their color changes per frame.

In a real-world setting, the most disentangled possible representation of stock market prices might involve a latent which represents a complex agglomeration of public opinion from the news, consumer confidence ratings, and estimates of the Fed's likelihood of raising rates. Such a latent might truly be the best and most independent axis of variation for predicting the stock price, yet it would not be as easy to interpret as a representation with one latent for public opinion, one latent for the Fed, and one latent for consumer confidence. In such a non-human domain, our intuitions about the factors of variation may not hold, and as a result the representations that make sense to us and those that accurately represent the factors of variation may diverge.

Interpretability is extremely valuable in many domains. If a doctor is attempting to plan a course of treatment for a patient, they need to be able to reason about the factors in a diagnostic model they're using. Even if an algorithm doesn't need to interface with a human at runtime, it's very hard to debug a system during development if you don't understand what it's doing.



## Performant

A _performant_ representation for a task contains the information needed to perform well on that task.

If the task is determining whether or not there is a dog in a room, a representation consisting of a photograph of the room would be less performant than a 3D voxel model of the room, which in turn would be less performant than a single binary bit representing whether or not there is a dog.


## Reusable

A _reusable_ representation is one that is performant for many tasks in the same domain.

To continue the example above, a 3D voxel representation of a room is more reusable than one indicating whether or not the room contains a dog. Somewhere in between the two would be a representation consisting of the facts,

- Is there an animal?
- Is it furry?
- How big is it?
- What color is it?

This representation would be able to solve the task of whether or not the room contains a dog with high probability, and would also be able to inform the task of whether or not the room contains a gorilla, or a whale. However, the tasks it can address are a strict subset of the voxel representation, which could also solve such problems as "Where is the couch?"


## Compact

A _compact_ representation is one which occupies few bits.

Compactness may not seem inherently important; it is typically irrelevant if the representation of an image takes up one megabyte or two. However, compactness provides a very valuable forcing function. One might build a weather forecasting model which represents the state of the world down to the very last butterfly.

The actions of this butterfly might be indeterminate given the other latents in the model, so it is disentangled; it might be perfectly easy to understand the meaning of the butterfly's representation, so it is interpretable; it might be valuable in some other system or context, so it is reusable; and it might even minutely improve the performance of the weather forecast, so it is performant. But somehow none of this quite justifies its presence in a weather model.

Compactness says that we only care about the most important factors of variation.
