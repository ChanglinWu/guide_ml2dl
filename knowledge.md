# The module we must use in ML/DL




## optimization algorithms

1. Stochastic Gradient Descent (SGD)
   Update Strategy: Update weights after each individual sample using its sample loss $L_i(\theta)$.When to use: Great for online learning, streaming data, or escaping local minima due to the noise in updates.
2. Batch Gradient Descent
   Update Strategy: Pass the entire dataset, compute the total dataset cost function $J(\theta)$, and make one update per epoch.When to use: Small datasets that easily fit in memory where stable, smooth convergence is preferred.
3. Mini-Batch Gradient Descent (Industry Standard)
   Update Strategy: Pass a small chunk of data (e.g., 32, 64, or 128 samples), compute the batch cost function, and update weights after each batch.When to use: Modern deep learning. It combines the speed and noise reduction of SGD with the hardware acceleration (GPU vectorization) of batch updates.

