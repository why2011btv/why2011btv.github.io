# linear regression duplicate data问系数、error、t-statistics、beta、R squared、standard error咋变？
If there are duplicate data in linear regression, the impacts on various components would be:

1. Coefficients (β): The estimates of the coefficients will not generally change, under the assumption that the duplicates are a perfect copy of the original observations.

2. Error: The residuals or errors would also simply be duplicated along with the dependent variable and hence, it may appear to reduce variability but does not affect the model fit.

3. T-statistics: T-statistics value will not change. However, if the duplicates substantially increase the sample size, they could artificially inflate the significance level of the model or predictors.

4. Beta: Beta coefficient or standardized coefficient would not change as they are calculated based on original data relationships.

5. R squared: R-squared value would remain the same, because the duplicate data will lie on the line of the regression model from original data.

6. Standard error: SE will get reduced. With duplication, the sample size increases and thereby reducing the standard error of the estimates. However, this paints an artificially confident picture of the estimates, which should be avoided. 

Overall, while duplicate data may distort some metrics, key parameters such as coefficients, R-squared will generally not alter. However, it isn't a sound statistical practice to knowingly include duplicate records in a regression analysis. Proper data cleaning should be carried out before model estimation.

# correlation coefficient和R squared的关系 
  - Correlation coefficient and R-squared are both statistical measures that are often used in regression analysis and they both describe the degree and the nature of the relationship between two or more variables.

1. Correlation Coefficient (r): This is a measure of the strength and direction of the linear relationship between two variables. It ranges from -1 to 1, with -1 indicating a strong negative relationship, 1 indicating a strong positive relationship, and 0 indicating no relationship at all.

2. R-squared (R²): Also known as the coefficient of determination, it is a statistical measure of how close the data are to the fitted regression line. It ranges from 0 to 1, with 0 indicating that none of the variance in the dependent variable is predictable from the independent variable(s), and 1 indicating that all of the variance in the dependent variable is predictable from the independent variable(s).

The relationship: The square of the correlation coefficient (i.e., r²) is equal to R-squared. R-squared measures the proportion of the variance in the dependent variable that can be predicted from the independent variable(s), so it is essentially a measure of the explanatory power of the regression model. Thus, a high r² means a more reliable linear regression prediction. If r is the correlation between X and Y, then r² = R², indicating that R² measures the square of the correlation between the observed and predicted values of the dependent variable.
# 如何detect collinearity
There are multiple methods for detecting collinearity (inter-correlations among predictor variables) in your dataset:

1. Correlation matrix: This method involves creating a matrix of Pearson's correlations between all pairs of variables. High correlation coefficients between pairs of variables may indicate a collinearity problem.

2. Variance Inflation Factors (VIF): A VIF quantifies how much the variance is inflated for each coefficient in the model. A VIF of 1 indicates no collinearity, while a VIF greater than 1 indicates some degree of collinearity. As a rule of thumb, a VIF larger than 5 (or sometimes 10) indicates a significant collinearity problem.

3. Condition Index: In a multiple regression analysis, a condition index above 15 indicates a potential problem with collinearity and above 30, a serious problem.

4. Eigenvalues: When conducting Principal Component Analysis, small or zero Eigenvalues are indicative of collinearity.

5. Tolerance: It is the reciprocal of the VIF. If the value of tolerance is less than 0.2 or 0.1, it’s generally considered an indication of high collinearity.

6. Scatter Plots and Correlation: Visualizing the data can sometimes help to check the pairwise relationships between variables. If two variables are found to have a linear relationship, then they might be collinear.

Properly identifying and addressing collinearity is crucial since it can impact the stability and interpretability of your regression model.# collinearity如何影响linear regression的variance
  - 1. High Variance: In presence of high collinearity, the estimates of parameters can have a substantial amount of variance, making them very sensitive to slight changes in the model. This makes the model unstable and the estimates unreliable, leading to overfitting.
# linear regression的assumptions
  - linearity
  - normality: The error terms (residuals of the model) are normally distributed. This can be checked by plotting a histogram of residuals and see if it forms a bell curve, or a Q-Q plot to see if it follows a straight line. 
  - constant variance
  - independence：The observations are independent of each other. This is important in the context of time series data where this assumption can be violated.
  - no serious multicollinearity: Linear regression assumes that predictors are not perfect linear functions of each other. To check for multicollinearity, we typically use Variance Inflation Factor (VIF).
