# My Little Red Book for Academic Writing
## A
- account
  - *noun*, a report or description of an event or experience.
  - e.g., *a detailed account of what has been achieved*
- aggravate
  - *verb*, make (a problem, injury, or offense) worse or more serious.
  - e.g., *The model's majority label bias is aggravated by its recency bias*
## C
- corroborate
  - *verb*, confirm or give support to (a statement, theory, or finding)
  - e.g., *Other works have corroborated some of these findings*
- conflate
  - *verb*, combine (two or more texts, ideas, etc.) into one.
  - e.g., *the urban crisis conflates a number of different economic and social issues*
## D


## E
- envisage
  - e.g., *spatial-temporal reasoning depends on envisaging the possibilities consistent with the relations between objects*

## F
## G
## H
## I 
- incidental
  - *adjective*, accompanying but not a major part of something.
  - e.g., *for the fieldworker who deals with real problems, paperwork is incidental*
## J
## K
## L
- lack behind
  - e.g., *However, current AI’s ability in “thinking in pictures” is still far lacking behind.*
- inter alia
  - e.g., *(Chambers and Jurafsky, 2008; Modi et al., 2017, inter alia)*
## M
## N
- narrative
  - *noun*, a spoken or written account of connected events; a story.


## R
- rectify
  - e.g., *Relabeling methods use the trained model to incrementally detect possibly noisy labels and rectify them with model predictions.*


## T
- trait
  - e.g., *This captivating commonplace trait of human cognition*



# Interview Questions
1. What is gradient descent? Write the formula of weight update.
  - Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model. 
  - ![bp_update_formula](https://user-images.githubusercontent.com/32129905/145055060-a3bf4742-73f1-4c21-9b3f-d2ee59aa3925.png)
2. What is gradient vanishing/exploding, and how to solve them? Sigmoid vs Tanh, which one would cause gradient vanishing more easily?
  - Backpropagation computes gradients by the chain rule. This has the effect of multiplying n of these small numbers to compute gradients of the early layers in an n-layer network, meaning that the gradient (error signal) decreases exponentially with n while the early layers train very slowly. And in some cases, the gradients keep on getting larger and larger as the backpropagation algorithm progresses. This, in turn, causes very large weight updates.
  - How to solve:
    - The simplest solution is to use other activation functions, such as ReLU, which doesn’t cause a small derivative.
    - Residual networks are another solution, as they provide residual connections straight to earlier layers. This residual connection doesn’t go through activation functions that “squashes” the derivatives, resulting in a higher overall derivative of the block.
    - Finally, batch normalization layers can also resolve the issue. It reduces this problem by simply normalizing the input so |x| doesn’t reach the outer edges of the sigmoid function and thus the derivative isn’t too small. Side note: BatchNorm makes the optimization landscape significantly smoother. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training ([Santurkar et al, NeurIPS'18](https://arxiv.org/abs/1805.11604)).
  - Since the hyperbolic tangent function has greater derivative over the sigmoid around zero, sigmoid causes gradient vanishing more easily.
3. How to overcome a local minimum problem?
  - Stochastic Gradient Descent. In stochastic gradient descent the parameters are estimated for every observation, as opposed the whole sample in regular gradient descent (batch gradient descent). This is what gives it a lot of randomness. The path of stochastic gradient descent wanders over more places, and thus is more likely to "jump out" of a local minimum, and find a global minimum. Note: It is common to keep the learning rate constant, in this case stochastic gradient descent does not converge; it just wanders around the same point. However, if the learning rate decreases over time, say, it is inversely related to number of iterations then stochastic gradient descent would converge.
  - SGD with momentum: An object that has motion (in this case it is the general direction that the optimization algorithm is moving) has some inertia which causes them to tend to move in the direction of motion. Thus, if the optimization algorithm is moving in a general direction, the momentum causes it to ‘resist’ changes in direction, which is what results in the dampening of oscillations for high curvature surfaces. To implement this momentum, exponentially weighted averages is used, which provides us a better estimate which is closer to the actual derivate than noisy calculations.
4. What is bias–variance tradeoff?
  - The bias-variance tradeoff refers to a decomposition of the prediction error in machine learning as the sum of a bias and a variance term.
  - Theoretical result: Test MSE = Bias^2 + Variance + Dataset label noise
  - "Bias" = how far the average fit is from the true function. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.
  - "Variance" = how different the different fits are (using different samples of training data). Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data.
  - ![1*RQ6ICt_FBSx6mkAsGVwx8g](https://user-images.githubusercontent.com/32129905/145097067-4fcee023-45e9-4327-8d1e-f5f15b3738ee.png)
5. Spilt the dataset into 10-fold vs 2-fold, which one yields model with higher variance? And which one yields model with higher bias?
  - 10-fold has higher variance; 2-fold has higher bias.
6. What is regularization?
  -  This technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. It significantly reduces the variance of the model, without substantial increase in its bias. Using L1 Norm (Lasso) or L2 Norm (Ridge) can achieve regularization.
7. What is the cost function of logistic regression?
  - ![1*CQpbokNStSnBDA9MdJWU_A](https://user-images.githubusercontent.com/32129905/145099928-3665389e-9b2d-44ce-96be-4e730fcb10d2.png)
8. What is ROC? AUC?
  - Can be used to DETERMINING THE OPTIMAL CUT-OFF VALUE FOR A TEST
  - ROC: summarizes all of the confusion matrices that each threshold produced. Y-axis: True Positive Rate = True Positive / (True Positive + False Negative); X-axis: False Positive Rate = False Positive / (False Positive + True Negative). How to construct: TPR against FPR under different threshold.
  - AUC: makes it easier to compare one ROC curve to another. Area under ROC curve. 0.9 means outstanding discrimination; 0.7-0.8: excellent; 0.6-0.7: Acceptable; 0.5-0.6: Poor; 0.5: Random guess
9. Label encoding vs One-hot encoding?
  - It can skew the estimation results if an algorithm is very sensitive to feature magnitude (like SVM). In such case you may consider standardizing or normalizing values after encoding.
  - It can skew the estimation results if there is a large number of unique categorical values. In our case it was 4, but if it’s 10 or more, you should keep this in mind. In such case you should look into other encoding techniques, for example, one hot encoding.
10. k-nearest neighbors
  - k=1 or 2 can be noisy and subject to the effects of outliers
  - you don't want k to be so large that a category with a few samples in it will always be out voted by other categories.
 11. K-means clustering
  - Specify number of clusters K.
  - Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
  - Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
    - Compute the sum of the squared distance between data points and all centroids.
    - Assign each data point to the closest cluster (centroid).
    - Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.
 12. Explain differences between linear and logistic regression. 
  - <img width="876" alt="Screen Shot 2021-12-08 at 9 09 28 PM" src="https://user-images.githubusercontent.com/32129905/145321542-21d7ce61-a8fd-4618-80f0-96ce9ad46ee9.png">
  - maximum likelihood estimation (MLE) is a method of estimating the parameters of an assumed probability distribution, given some observed data. 
 
