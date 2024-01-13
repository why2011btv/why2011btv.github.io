# CIS 520 Review
1. SVD, left single vector, eigenvector
2. GAN
3. HMM
4. GMM
5. Q-learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/intro-sarsa/
6. EM: https://blog.csdn.net/zouxy09/article/details/8537620
7. Bayesian Belief Networks: https://blog.csdn.net/bluebelfast/article/details/51509223
    - 先简单总结下频率派与贝叶斯派各自不同的思考方式：
    - 频率派把需要推断的参数θ看做是固定的未知常数，即概率虽然是未知的，但最起码是确定的一个值，同时，样本X 是随机的，所以频率派重点研究样本空间，大部分的概率计算都是针对样本X 的分布；
    - 而贝叶斯派的观点则截然相反，他们认为参数是随机变量，而样本X 是固定的，由于样本是固定的，所以他们重点研究的是参数的分布。
    - <img width="1609" alt="Screen Shot 2021-12-15 at 12 39 05 PM" src="https://user-images.githubusercontent.com/32129905/146237095-019dbe28-48df-4a85-93ed-7f5e693def85.png">

    
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
  - A process that changes the result answer to be "simpler": L1 & L2 regularization; dropout; cross-validation
  - This technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. It significantly reduces the variance of the model, without substantial increase in its bias. Using L1 Norm (Lasso) or L2 Norm (Ridge) can achieve regularization.
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
  - The Similarities between Linear Regression and Logistic Regression
    - Linear Regression and Logistic Regression both are supervised Machine Learning algorithms.
    - Linear Regression and Logistic Regression, both the models are parametric regression i.e. both the models use linear equations for predictions
  - The Differences between Linear Regression and Logistic Regression
    - Linear Regression is used to handle regression problems whereas Logistic regression is used to handle the classification problems.
    - Linear regression provides a continuous output but Logistic regression provides discreet output.
    - The purpose of Linear Regression is to find the best-fitted line while Logistic regression is one step ahead and fitting the line values to the sigmoid curve.
    - The method for calculating loss function in linear regression is the mean squared error whereas for logistic regression it is maximum likelihood estimation.
  - maximum likelihood estimation (MLE) is a method of estimating the parameters of an assumed probability distribution, given some observed data. 
13. There are 3 main metrics for model evaluation in regression:
  - R Square/Adjusted R Square
  - Mean Square Error(MSE)/Root Mean Square Error(RMSE)
  - Mean Absolute Error(MAE)
14. What’s the equivalent of R^2 in logistic regression? Pseudo R^2: log-linear ratio R^2; Cox and Snell's R^2; Nagelkerke’s R2
15. What’re the common metrics for evaluating logistic regression models? Accuracy, Precision, Recall, Confusion Matrix
16. Should we rescale features before gradient descent? Why?
  - If an algorithm uses gradient descent, then the difference in ranges of features will cause different step sizes for each feature. To ensure that the gradient descent moves smoothly towards the minima and that the steps for gradient descent are updated at the same rate for all the features, we scale the data before feeding it to the model. Having features on a similar scale will help the gradient descent converge more quickly towards the minima.
  - Specifically, in the case of Neural Networks Algorithms, feature scaling benefits optimization by:
    - It makes the training faster
    - It prevents the optimization from getting stuck in local optima
    - It gives a better error surface shape
    - Weight decay (lambda for L2-regularization) and Bayes optimization can be done more conveniently
17. Advantages of decision trees:
  - Are simple to understand and interpret. People are able to understand decision tree models after a brief explanation.
  - Have value even with little hard data. Important insights can be generated based on experts describing a situation (its alternatives, probabilities, and costs) and their preferences for outcomes.
  - Help determine worst, best, and expected values for different scenarios.
  - Use a white box model. If a given result is provided by a model.
18. Transformer
  - Transformer architecture ditched the recurrence mechanism in favor of multi-head self-attention mechanism => FASTER processing, can attend to long-distance context; 1-D convolutions don't have memory, and it only processes local context
  - positional encoding: As each word in a sentence simultaneously flows through the Transformer’s encoder/decoder stack, The model itself doesn’t have any sense of position/order for each word. Consequently, there’s still the need for a way to incorporate the order of the words into our model => Transformers hold the potential to understand the relationshipbetween sequential elements that are far from each other.
  - Transformers serve to be helpful in anomaly detection.
  - <img width="420" alt="Screen Shot 2021-12-09 at 7 06 14 AM" src="https://user-images.githubusercontent.com/32129905/145393526-4d3dfe1e-7d86-4094-beac-405d2839da9a.png">
  - Self-attention:
    - far-away inputs can influence output at t
    - influence no longer just a function of relative position s-t
    - input transformed first through f(.)

19. Word2Vec How is it trained? Loss function?
  - <img width="717" alt="Screen Shot 2021-12-08 at 9 50 10 PM" src="https://user-images.githubusercontent.com/32129905/145325601-26ce67ac-1496-41ed-95b4-2be2c73b1ac1.png">
  - Essentially, this is a multi-class classification problem. Outputs are one-hot vectors. Loss: standard softmax loss.
  - Instead of summing over the size of vocabulary, randomly sampling to approximate the denominator (since the correct value of the numerator should be a lot higher than the other values, particularly it should be higher than random samples). In this way, fewer weights need to be updated during each iteration of gradient descent.
  - Word2Vec representation is fixed, so it is not contextual in usage. The representation for a multi-meaning word is the same under different context.
  - CBOW: predicts central word, basing on surrounding words.
  - Skip-Gram: indicates the context (surrounding words) using selected, single words.
20. PCA
  - reduce the dimension of your feature space.
22. SVD
23. LDA
24. Boosting
  - ![0*6AINiXpqj_O98Kf-](https://user-images.githubusercontent.com/32129905/145491326-e3bc390e-76f1-46be-8bb1-7fb87d69893a.png)
  - weighted majority vote
  - models with low error are weighted more than those with high error
  - reweighting the dataset
26. AdaBoost: It iteratively adds classifiers, each time reweighting the dataset to focus the next classifier on where the current set makes errors.
27. Bagging: short for "bootstrap aggregating", yields smaller variance
  - bootstrap: random sampling with replacement 
  - ![0*Wjdc5fBd53V108Qn](https://user-images.githubusercontent.com/32129905/145491070-95d2d97b-c10c-4514-b0fc-655c0621660b.png)
28. Boosting vs Bagging
  - sampling method
    - Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。
    - Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。
  - sample weights
    - Bagging：使用均匀取样，每个样例的权重相等
    - Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。
  - models
    - Bagging：所有预测函数的权重相等。
    - Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。
  - parallel or sequential
    - Bagging：各个预测函数可以并行生成
    - Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。
28. Random Forests: an ensemble method that combines decision trees
  - Pros: Excellent Predictive Power; Interpretability; No normalization required; Fine with missing data
  - Cons: Parameter Complexity and Overfitting Risk; Limited with regression especially when data has linear nature; Biased towards variables with more levels
30. Ensemble
31. Hypothesis Set
    - The hypothesis set 𝐻 is the set of all candidate formulas (or candidate models if you like) that could possibly explain the training examples we have.
    - Our learning algorithm 𝐴 (that being a straightforward learning routine like linear regression or an elaborate learning routine like a gradient boosting machine) allows us to make the optimal choice of ℎ∈𝐻 that the algorithm 𝐴 produces. Notice that the hypothesis test 𝐻 is related the learning algorithm 𝐴. For example, a linear regression can only "learn" linear models (if we do not incorporate interactions) while a gradient boosting machine can learn non-linear relations more easily.
32. Inductive Bias
    - the set of assumptions that the learner uses to predict outputs of given **inputs that it has not encountered**
    - The following is a list of common inductive biases in machine learning algorithms.
        - Maximum conditional independence: if the hypothesis can be cast in a Bayesian framework, try to maximize conditional independence. This is the bias used in the Naive Bayes classifier.
        - Minimum cross-validation error: when trying to choose among hypotheses, select the hypothesis with the lowest cross-validation error. Although cross-validation may seem to be free of bias, the "no free lunch" theorems show that cross-validation must be biased.
        - Maximum margin: when drawing a boundary between two classes, attempt to maximize the width of the boundary. This is the bias used in support vector machines. The assumption is that distinct classes tend to be separated by wide boundaries.
        - Minimum description length: when forming a hypothesis, attempt to minimize the length of the description of the hypothesis.
        - Minimum features: unless there is good evidence that a feature is useful, it should be deleted. This is the assumption behind feature selection algorithms.
        - Nearest neighbors: assume that most of the cases in a small neighborhood in feature space belong to the same class. Given a case for which the class is unknown, guess that it belongs to the same class as the majority in its immediate neighborhood. This is the bias used in the k-nearest neighbors algorithm. The assumption is that cases that are near each other tend to belong to the same class.
