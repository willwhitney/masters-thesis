## Experiments

In order to carefully test the ability of various techniques to correctly factorize several presented problems, I constructed a simple dataset of vector functions, inputs, and outputs. These functions are detailed in [@Tbl:primitives]. In the following experiments, these functions are applied to random input vectors in $[0,1]^{10}$.

Since the inputs to all of these functions are indistinguishable, without any extra information it would be impossible for any system to achieve results better than averaging the output of all these functions. Therefor, along with the input vector, all systems receive a one-hot vector containing the index of the primitive to compute. Each system must learn to interpret this information in its own way, and there is no hand-designed routing of 


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

: **Table of primitive functions.** The true test of a learned model is how distinctly the model manages to represent these functions, not the exact error number. Outputs shown for the input vector [1 2 3 4 5 6 7 8]. {#tbl:primitives}

### Disentanglement of functions

![**Disentanglement and validation loss** plotted over the course of training. Disentanglement, or _independence_, is measured by the L2 norm of the weight vector over the functions. In this measure, 0.35 is totally entangled, with every function accorded equal weight for every input, and 1.0 is totally disentangled, with precisely one function used for each input. **Left:** with sharpening and noise. **Right:** without sharpening and noise.](../figures/combo-loss-entanglement.png){#fig:loss_entanglement}

### Catastrophic forgetting

![**Forgetting when trained on one task.** When a traditional feedforward network, which previously trained on several tasks, is trained exclusively on one, it forgets how to perform the others. The controller-function network is practically immune to forgetting. In this figure, we see each network trained exclusively on one of several tasks it is able to do. The loss that is shown is the average L2 error attained on all of the _other_ tasks as this network retrains.](../figures/forgetting.png){#fig:forgetting}
