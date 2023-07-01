
Softmax with Cross Entropy Loss 

# Cross Entropy Loss function
- loss/cost functions are used to optimize the model during training.
- The objective is almost always to **minimize the loss function**
- Cross-Entropy loss is used to optimize **classification models**
- Cross-Entropy loss and Softmax are pegged.
- In the basic, the loss function will take the output of forward and measure the distance from the truth valuse    
- Math: LOSS(yi,pi) = -SUM: yi * log(pi), for n classes
  - yi is the true label
  - pi is the softmax probability or the i class

## Binary Cross-Entropy Loss
- Two classes, 1 or 0.
- Math: L = -SUM: yi * log(pi)
  - yi is the true value, 0 or 1
  - pi is the softmax probability for the i class
- Is often calculated as the average cross-entropy across all data examples
- Math: L = - 1/N * [SUM: [yi * log(pj) + (1-yi) * log(1 - pj)]]
  - for N data points where yi is the truth value taking a value 0 or 1 and pi is the softmax probability for the i data point

## Categorical Cross-Entropy Loss && Sparse Categorical Cross-Entropy Loss
- are equal, the only diference is on how truth labels are defined
  - Categorical cross-entropy is used when tru labels are **one-hot encoded**, example, we have the following true values for 3-class classification problem [1,0,0],[0,1,0] and [0,0,1]
  - Sparse categorical cross-entropy, truth labels are integer encoded, for example, [1],[2] and [3] for 3-class problem


# References
https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e



