# Lecture 2: Linear Regression and Gradient
## Info
    Notes by: Rahmani Dibansa
    Github: https://github.com/RemainAplomb
    Retrieved from: https://www.youtube.com/watch?v=4b4MUYve_U8&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=3


## Motivate Linear Regression
    Definition: In Regression, our output/groundtruth (Y) is continous

    Example: Supervised learning for simple automatic driving. To train, we
        give an input(X) which is the grayscaled image of the path ahead,
        and an output/groundtruth(Y) which is the direction of the steering wheel.


## Supervised Learning
    Definition: We are give an X that will be mapped to an output Y

    Example: Regression Supervised learning for simple automatic driving. To train, 
        we give an input (X) which is the grayscaled image of the path ahead,
        and an output/groundtruth (Y) which is the direction of the steering wheel.
    
    Example: We have a dataset which contains the size of the house and its pricee.
        And we want to use it to train a model that will predict the price of the house
        depending on its size


## Designing a Learning Algorithm
    Linear Algebra:

    Considerations for structuring a machine learning algorithm:
        - What is the dataset?
        - What is the hypothesis?
        - How does it represent the hypothesis?
    
    How do you represent the hypothesis (H)?
        - h(x) = Θ0 + Θ1x
        - We have a function of x that represents a linear function
          with a size x.
        - We have our input x, but we can also have multiple input
          information. Like size of the house, and number of beds
        - If we have multiple inputs. We can represent them as:
            X1 : Size of the house
            X2 : Number of bedrooms
            h(x) = Θ0 + ((Θ0)(X1)) + ((Θ2)(X2))
    
    Simplfying the notation:
        - The hypothesis as sum from the J equals 0-2 of theta JXJ
        - Where for conciseness, we define X0 to be equal to 1
        - If we define X0 to be a dummy feature that always takes on
          the value of 1, then we can write the hypothesis h of x this way
        - Theta (Θ) becomes a three-dimensional parameter [Theta0, Theta1, Theta2]
        - The features (X) becomes a three-dimensional vector [X0, X1, X2]
        - Theta(Θ) is called the parameters of the learning algorithm. In the previous 
          example, the job of the learning algorithm is to choose parameters theta, 
          that allows us to make good predictions about the prices of houses


## Parameters of Learning Algorithm
    Definition: Theta(Θ) is called the parameters of the learning algorithm. In the previous 
        example, the job of the learning algorithm is to choose parameters theta, that allows 
        us to make good predictions about the prices of houses

    Defining notations:
        - m : the number of training examples (the number of rows in the training set)
        - x : inputs/features
        - y : outputs/target variable/groundtruth
        - (x, y) : a training example
        - ((x_i), (y_i)) : denotes the i_th training example
        - n : number of features. (In the previous example, n = 2. The size, and number of bedrooms)
    
    Goal of the learning algorithm:
        - The learning algorithm's job is to choose values for the parameters theta so that
          it can output a hypothesis/prediction
        - Choose theta such that h of x is close to y for the training examples
        - Sometimes we emphasize that h depends on the parameters theta and on the input
          features x. It goes like this: h_Θ(x) = h(x)
        - In previous example, we choose theta so that the output of the learning algorithm is close
          to where the correct price is for that set of houses


## Linear Regression Algorithm
    In linear regression algorithm, we want to:
        - Minimize the squared difference between the prediction and the correct price(h_Θ(x) -y)^2
        - Choose the value of theta so that difference can be minimized
        - J(Θ) = ... (I cant understand this)

## Cost Function
    In linear regression, the cost function (also known as the loss function or the objective function) measures the error between the predicted values (hypothesis) and the actual values (ground truth) of the target variable. The goal is to minimize this error to find the optimal parameters (Thetas) for the hypothesis.

    In the case of linear regression, the commonly used cost function is the Mean Squared Error (MSE). The MSE calculates the average squared difference between the predicted values (hypothesis) and the actual values (ground truth) over the entire training dataset.

    The formula for the MSE cost function is:

    J(Θ) = (1/2m) * Σ[(hΘ(x^(i))) - y^(i)]^2

    Let's break down the components of the formula:

    J(Θ): Represents the cost function. It measures the overall error of the hypothesis for a given set of parameters Θ.
    m: Represents the number of training examples in the dataset.
    hΘ(x^(i)): Represents the predicted value (hypothesis) for the i-th training example, obtained using the current parameters Θ.
    y^(i): Represents the actual value (ground truth) for the i-th training example.
    The goal of the linear regression algorithm is to find the parameters Θ that minimize the cost function J(Θ). This minimization process is achieved using an optimization algorithm called gradient descent.


## Gradient Descent and Parameter Update
    Gradient descent is an iterative optimization algorithm used to minimize the cost function by adjusting the parameters (Thetas) in each iteration. The basic idea is to take steps in the direction of the steepest descent of the cost function to eventually reach the minimum.

    The update rule for gradient descent in linear regression is as follows:

    Θ_j := Θ_j - α * (1/m) * Σ[(hΘ(x^(i))) - y^(i)] * x_j^(i)

    Let's explain the components of the update rule:

    Θ_j: Represents the j-th parameter (weight) that needs to be updated.
    α (alpha): Represents the learning rate, which determines the step size of each update. It controls how quickly the algorithm converges to the minimum. It is a hyperparameter that needs to be tuned.
    m: Represents the number of training examples in the dataset.
    hΘ(x^(i)): Represents the predicted value (hypothesis) for the i-th training example, obtained using the current parameters Θ.
    y^(i): Represents the actual value (ground truth) for the i-th training example.
    x_j^(i): Represents the j-th feature value of the i-th training example.
    In each iteration of gradient descent, the update rule is applied to update all the parameters Θ_j simultaneously. This process continues until the algorithm converges or reaches a stopping criterion (e.g., maximum number of iterations).

    By iteratively updating the parameters using the gradient descent algorithm, the cost function gradually decreases, leading to better-fitted parameters that minimize the error between the predicted values and the actual values.

## Gradient Descent
    Definition:
        - We start with some value of theta (Θ), and keep changing it so that we can reduce J(Θ).
        - We start of in some point in the gradient descent, imagine we are standing on some hill.
          What we do in gradient descent is to turn 360 and look around to see where we can take a 
          tiny step in a direction that will help us go downhill as fast as possible. We want to 
          go to the lowest possible point/elavation of J of theta
    
    A step of gradient descent can be implemented as follows:
        - Theta J gets updated as Theta J minus alpha (learning rate) multiplied by the partial
          derivative of the cost function J with respect to the parameter theta J.....
    
    How the derivative calculation is done? (Around 27:30 in the youtube video)
        - .... (I cant understand this)
      
    We will let theta J be updated according to Theta J minus the learning rate multipiled by
    h of x minus y times XJ

    For the m (number of examples), we sum the derivatives over all the m training examples.

    The derivative of a sum is the sum of the derivatives (Would be great if this can be expanded)


## Derivative Calculation
    The derivative calculation is an important step in the gradient descent algorithm. It involves computing the partial derivatives of the cost function with respect to each parameter (Theta) in order to determine the direction and magnitude of the updates.

    Let's walk through the process of calculating the derivatives for the linear regression cost function.

    Recall the cost function for linear regression:

    J(Θ) = (1/2m) * Σ[(hΘ(x^(i))) - y^(i)]^2

    To calculate the partial derivative of J(Θ) with respect to a specific parameter Θ_j, we can follow these steps:

    Expand the cost function:
    J(Θ) = (1/2m) * Σ[(Θ_0 * X_0^(i) + Θ_1 * X_1^(i) + ... + Θ_j * X_j^(i) + ... + Θ_n * X_n^(i)) - y^(i)]^2

    Here, X_0^(i) is the dummy feature equal to 1 for all training examples.

    Take the derivative with respect to Θ_j:
    To calculate the derivative with respect to Θ_j, we differentiate the squared term inside the summation:

    ∂J(Θ)/∂Θ_j = (1/m) * Σ[(Θ_0 * X_0^(i) + Θ_1 * X_1^(i) + ... + Θ_j * X_j^(i) + ... + Θ_n * X_n^(i)) - y^(i)] * X_j^(i)

    In this expression, ∂J(Θ)/∂Θ_j represents the partial derivative of the cost function J(Θ) with respect to Θ_j.

    Simplify the expression:
    You can simplify the expression further by rearranging the terms:

    ∂J(Θ)/∂Θ_j = (1/m) * Σ[(hΘ(x^(i))) - y^(i)] * X_j^(i)

    In this form, (hΘ(x^(i))) represents the predicted value (hypothesis) for the i-th training example.

    Sum over all training examples:
    To account for all training examples, you sum up the derivatives over the entire training set:

    ∂J(Θ)/∂Θ_j = (1/m) * Σ[(hΘ(x^(i))) - y^(i)] * X_j^(i)

    Finally, you repeat this process for each parameter Θ_j to obtain the partial derivatives for all parameters.

    The partial derivatives determine the direction of the steepest descent, indicating how the parameters should be updated in order to minimize the cost function. By using these derivatives in the gradient descent update rule, the algorithm adjusts the parameters iteratively until convergence is reached.



## Gradient Descent Algorithm
    Definition:
        - Is to repeat taking steps in the gradient descent until convergence. And in each iteration of
          gradient descent, you do this for j = 0, 1, ..., n
        - If you do this, what will happen is you hopefully find a good value of theta.
    
    When we plot the cost function j of theta for a linear regression model, unlike the earlier diagrams with local optima, it turns out that if j of theta is defined the way that is the sum of squared terms, then j of theta is a quadratic function, right? It's a sum of these squares of terms. The j of theta will always look like a big bowl

    J of theta does not have many local optimas, instead it only have the local optima as the global optimum.

    If we think of the contours of the functions, the steepest descent is always at 90 degrees, it is always orthogonal to the contour direction

    If you set the alpha/learning rate to be too large, then it can overshoot. This means that each step 
    is updating too much which results to it not finding the global optimum.

    What we usually do is we keep on changing the value of the learning rate so that we can find the most efficient way to drive down the value of j of theta

    Tip: Reduce the learning rate exponentially when you are just trying to figure things out. Example 1-e2,
      1-e4, 1-e8, 1-e16
    
    In the previous example, at the start, the hypothesis will be equal to 0, and if we graph it, we will get
    a horizontal line. However, as we train it, and update the value of the theta in each iteration, we will
    lower the squared errors. Eventually, the hypothesis will converge.
    

## Batch Gradient Descent
    Definition:
        - It refers to the entire training set, and we process the data as a batch. Exampl, we have a training
          dataset with 49 examples we train them all as a batch. We read them before we make an update/step
        - The disadvantage of this is that when we have a large training dataset, suppose 1 million examples,
          then, we have to look through all examples before making a tiny step. This is too slow, and expensive
    
    Batch gradient descent refers to processing the entire training set (batch) to update the parameters (Thetas) in each iteration.

    It involves calculating the gradients (derivatives) of the cost function with respect to the parameters using the entire dataset.

    The advantage of batch gradient descent is that it provides a more accurate estimation of the gradients as it considers the complete training set.

    However, it can be computationally expensive and slow, especially when dealing with large datasets. The algorithm needs to process all examples before making an update.


## Stochastic Gradient Descent
    Definition:
        - Instead of scanning through all of the examples in the training dataset before updating the 
          parameters theta, in stochatic gradient descent, we loop through i = 1 to m.
        - In the previous example, if we use stochastic gradient descent, we will scan at each houses one a 
          time, and after each scan, we will modify the parameters (theta)
        - As we run stochastic gradient descent, it will never quite converge. Instead, it will oscillate near
          the global minima
        - If we have a large dataset, the stochastic gradient descent becomes the best choice since this
          algorithm allows us to make faster progress
    
    For training our models, usually, we don't really need to get to the global minima. Being near it is enough
    As such, when using stochastic gradient descent is good enough even though it never really reaches the 
    global minima.

    In linear regression, we don't have local optimum, and in this cases it is easier to train since we won't
    have massive convergence problems. However, when we are training highly non-linear things like neural networks, convergence problems becomes more acute.

    If you have a small dataset, and it doesn't cost much to use batch gradient descent, then it is great to go
    for it.

    Stochastic gradient descent (SGD) updates the parameters after processing each individual training example.

    In each iteration, it randomly selects one training example and computes the gradients based on that example only.

    The main advantage of SGD is its computational efficiency, especially for large datasets. It allows for faster progress as each step involves processing only one example.

    However, SGD may exhibit more oscillations and noisy updates compared to batch gradient descent.

    SGD is particularly useful in scenarios where the training set is large, and the algorithm can converge without the need for precisely reaching the global minimum. It often provides good enough results.


## Choosing Between Batch Gradient Descent and Stochastic Gradient Descent
    The choice between batch gradient descent and stochastic gradient descent depends on various factors:
      - Dataset size: If you have a small dataset, batch gradient descent may be feasible since the computational cost is manageable. It provides more stable updates and can converge to the global minimum.

      - Dataset complexity: If the dataset is highly non-linear or involves training deep neural networks, stochastic gradient descent is often preferred due to faster progress and less sensitivity to local optima.

      - Computational resources: If you have limited computational resources or dealing with large datasets, stochastic gradient descent becomes more favorable due to its efficiency.

      - Convergence requirements: If you require precise convergence to the global minimum, batch gradient descent may be more appropriate. However, if near-optimal results are sufficient, stochastic gradient descent can be a good choice.

      - In practice, variations of gradient descent, such as mini-batch gradient descent, are often used. Mini-batch gradient descent processes a small batch of training examples (between one and the entire training set) to strike a balance between the stability of batch gradient descent and the efficiency of stochastic gradient descent.

      - Ultimately, the selection of the gradient descent variant depends on the specific problem, available resources, and desired trade-offs between computational efficiency and convergence precision.



## Normal Equation
    The normal equation, also known as the closed-form solution, is an alternative method for finding the optimal parameters in linear regression without using iterative optimization algorithms like gradient descent. The normal equation directly computes the solution by solving a system of linear equations.

    The normal equation for linear regression is as follows:

    Θ = (X^T * X)^(-1) * X^T * y

    Here, Θ represents the vector of optimal parameters (Thetas), X is the design matrix that includes all the input features (including the dummy feature X_0 = 1), y is the vector of target values (ground truth), and (^T) denotes matrix transpose.

    To apply the normal equation, the design matrix X must be of full rank, which means all the features are linearly independent. If the design matrix is rank-deficient (some features are linearly dependent), the inverse term (X^T * X)^(-1) may not exist. In such cases, techniques like regularization can be used to address the issue.

    The normal equation offers some advantages:

    No need for manual selection of learning rate (alpha): Unlike gradient descent, the normal equation directly computes the optimal parameters without requiring a learning rate or iterations.

    Direct solution: The normal equation provides an exact solution without the need for iterative updates, making it computationally efficient, especially for small to medium-sized datasets.

    However, there are a few considerations and limitations:
        - Computational complexity: Computing the inverse term (X^T * X)^(-1) has a time complexity of O(n^3), where n is the number of features. Inverting large matrices can be computationally expensive, especially when the number of features is large.

        - Memory requirements: Inverting the (X^T * X) matrix requires storing the entire dataset in memory, which can be an issue for large datasets.

        - Non-invertibility: The (X^T * X) matrix may not be invertible if the features are not linearly independent. In such cases, the normal equation cannot be applied, and alternative methods like regularization (e.g., ridge regression, lasso regression) may be used.

    In summary, the normal equation provides a direct solution to linear regression without the need for iterative optimization. It can be a suitable approach for small to medium-sized datasets with linearly independent features. However, for larger datasets or when features are linearly dependent, iterative methods like gradient descent or regularization techniques may be more practical.


## Trace of A/Matrix
    The trace of a square matrix is the sum of its diagonal elements. In the context of linear algebra and machine learning, the trace of a matrix can have various applications and interpretations.

    In the lecture, the trace of a matrix was mentioned in the context of regularization and the cost function. Let's explore its role in regularization.

    Regularization is a technique used in machine learning to prevent overfitting and improve the generalization ability of a model. One common form of regularization in linear regression is called Ridge regression, which adds a regularization term to the cost function.

    The Ridge regression cost function is defined as follows:
        J(Θ) = (1/2m) * [(h(Θ)(x^(i)) - y^(i))^2] + (lambda/2m) * ∑(Θ_j)^2

    In this equation, the first term represents the mean squared error, similar to the standard linear regression cost function. The second term is the regularization term, where lambda (λ) is the regularization parameter that controls the amount of regularization applied.

    The regularization term involves the sum of the squared values of the parameters Θ_j (excluding the bias term Θ_0). This sum of squares can be calculated using the trace of the matrix Θ^TΘ, where Θ is the parameter matrix.

    Specifically, we have:
        ∑(Θ_j)^2 = trace(Θ^TΘ)

    The trace of a matrix is equal to the sum of its eigenvalues, which provides a measure of the sum of squares of the matrix elements. By using the trace, we can calculate the regularization term efficiently.

    Intuitively, the regularization term penalizes large values of the parameters Θ_j. This helps to prevent overfitting by discouraging the model from relying too heavily on any single feature and promoting a more balanced solution.

    The regularization parameter lambda (λ) controls the strength of regularization. A higher value of lambda increases the penalty for large parameter values, leading to more regularization and potentially simpler models. A lower value of lambda reduces the regularization effect, allowing the model to fit the training data more closely but increasing the risk of overfitting.

    In summary, the trace of a matrix is used in regularization techniques, such as Ridge regression, to compute the sum of squared parameters. By incorporating the trace in the cost function, regularization helps control overfitting and improves the generalization performance of the model.


# Summary

    We define a cost function based on sum of squared errors. The job is minimise this cost function with respect to the parameters. First, we look at (Batch) gradient descent. Second, we look at Stochastic gradient descent, which does not give us the exact value at which the minima is achieved, however, it is much much more effective in dealing with big data. Third, we look at the normal equation. This equation directly gives us the value at which minima is achieved! Linear regression models is one of the few models in which such an equation exist.