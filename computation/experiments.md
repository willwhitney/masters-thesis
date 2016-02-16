## Experiments

In order to carefully test the ability of various techniques to correctly factorize several presented problems, I constructed a simple dataset of vector functions, inputs, and outputs. These functions are detailed in [@Tbl:primitives]. In the following experiments, these functions are applied to random input vectors in $[0,1]^{10}$.

Since the inputs to all of these functions are indistinguishable, without any extra information it would be impossible for any system to achieve results better than averaging the output of all these functions. Therefore, along with the input vector, all systems receive a one-hot vector containing the index of the primitive to compute. Each system must learn to interpret this information in its own way. In the CFN, this metadata is passed only to the controller, which forces it to use different functions for different inputs.

While this is on the surface a supervised learning task (given some input, produce exactly this output), the much more interesting interpretation of the task is unsupervised. The true goal of this task is to learn a _representation of computation_ which mirrors the true factorization of the functions which generated this data. If we are interested in disentangled representations, we should look for systems which activate very distinct units for each of these separate computations.


------------------------------------------
Operation	Description										Output
---------	-----------										-------------------
rotate		Move each element of the vector right one slot.	[8 1 2 3 4 5 6 7]
			Move the last component to the first position.

add-a-b		Add the second half of the vector to the first.	[6 8 10 12 5 6 7 8]

rot-a		Rotate only the first half of the vector.		[4 1 2 3 5 6 7 8]

switch		Switch the positions of the first and second	[5 6 7 8 1 2 3 4]
			halves of the vector.

zero		Return the zero vector.							[0 0 0 0 0 0 0 0]

zero-a		Zero only the first half of the vector.			[0 0 0 0 5 6 7 8]

add-one		Add 1 to the vector.							[2 3 4 5 6 7 8 9]

swap-first	Swap the first two elements of the vector.		[2 1 3 4 5 6 7 8]

-----------------------------------

: **Primitive functions.** The true test of a learned model is how distinctly the model manages to represent these functions, not the exact error number. Outputs shown for the input vector [1 2 3 4 5 6 7 8]. {#tbl:primitives}

### Disentanglement of functions

![**Disentanglement and validation loss** plotted over the course of training. Disentanglement, or _independence_, is measured by the L2 norm of the weight vector over the functions. In this measure, 0.35 is totally entangled, with every function accorded equal weight for every input, and 1.0 is totally disentangled, with precisely one function used for each input. **Left:** with sharpening and noise. **Right:** without sharpening and noise.](../figures/combo-loss-entanglement.png){#fig:loss_entanglement}

In order to directly test how disentangled the CFN's representations are, I analyzed the weights given to each function in the network throughout the training process. In the ideal case, the distribution would be entirely concentrated on one function at a time; this would indicate that the network has perfectly decoupled their functions. Since no two functions are the same, and they each have the same input domain, no one function layer can correctly compute two of them.

The results of this analysis are presented in [@Fig:loss_entanglement]. By using the continuation method described in [@Sec:continuation], the CFN is able to very rapidly learn a disentangled representation of the functions in the data with no penalty to performance. By comparison, a network of the same architecture trained without the noise and sharpening technique can also produce the same output, but its representation of the computation is very highly entangled.



### Catastrophic forgetting

![**Forgetting when trained on one task.** When a traditional feedforward network, which previously trained on several tasks, is trained exclusively on one, it forgets how to perform the others. The controller-function network is practically immune to forgetting. In this figure, we see each network trained exclusively on one of several tasks it is able to do. The loss that is shown is the average L2 error attained on all of the _other_ tasks as this network retrains.](../figures/forgetting.png){#fig:forgetting}

To test the CFN's resistance to the forgetting problems which have plagued deep methods, I trained a controller-function network and a feedforward network to convergence on the full dataset, including all eight vector functions. The feedforward network was densely connected, with three linear layers of dimension 18, 100, and 10, with PReLu non-saturating activations in between.

After training each of these networks on the full dataset, I then set them to training on a dataset consisting of data from only one of the primitive functions. Both networks were retrained with the same learning rate and other hyperparameters. Periodically, I evaluated both networks' performance against a validation set consisting of data generated by all of the _other_ functions. As depicted in [@Fig:forgetting], the feedforward neural network experienced increasing loss over the course of training. These results are typical of neural methods. By contrast, the controller-function network has practically no forgetting behavior at all; the controller is assigning near-zero weights to all functions except the correct one, and as a result they receive gradients very near zero and do not noticeably update.

This result is especially compelling given the difference in parameter dimensionality of these two models; while the feedforward network has 13013 parameters, the CFN has better performance and better resistance to forgetting with only 2176. Though feedforward models with fewer parameters have worse forgetting behavior, the structure of the CFN representation allows for a very good memory.
