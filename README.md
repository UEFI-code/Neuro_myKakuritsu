# Neuro myKakuritsu Research Code with PyTorch

You may heard of [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) which is A Simple Way to Prevent Neural Networks from Overfitting.

While, the author thought the neuro cells died in brain is useful, because it make neuros have ability to random cooperation and can prevent overfitting during learning.

We think currently computers can only simulate fewer neuros, the random death of neuros also cause serious memory lossing, makes convergence harder. The key reason that cause overfit is dataset's problem, enhance the dataset is the best way, but improving dataset is hard work. 

To increasing the Neuro Network's performance with limited dataset, We can increase the single neuro level Divergent ability, with myKakuritsu Activation.

Kakuritsu means probability in English, instead of killing neuro cell, We let each synapse activation with a probability. This will make hidden layer's data Generalization during training but less memory lossing, maybe prevent overfitting better.

Still Need more Experment to prove this guess.
