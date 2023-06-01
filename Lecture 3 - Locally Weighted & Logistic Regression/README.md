# Lecture 3: Locally Weighted & Logistic Regression

## Info

    Notes by: Rahmani Dibansa
    Github: https://github.com/RemainAplomb
    Retrieved from: https://www.youtube.com/watch?v=het9HFqo1TQ&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=3

## Locally weighted regression

    Locally weighted regression is a non-parametric learning algorithm.
    This means that the amount of parameters will grow linearly
    along with the size of the training set

    In comparison, in parametric learning algorithms, no matter how big
    your training set is, you can erase the training set parameter and only
    have the parameter for theta i to make predictions.

    Parametric learning algorithms has a fixed set of parameters to the data.

    In non-parametric learning algorithms, the amount of paramters that you have to keep will grow linearly with the size of the training set. As such, Non-parametric learning algorithms will not be great if you have a massive training set, because you will need a lot of computer memory just to store all the parameters

    To evaluate h at a certain x:
    In linear regression, you fit theta to minimize the cost function that was provided.
    And it returns theta transpose x.
    On the other hand, for locally weighted regression, if we want to make a prediction at a value of x, we will have to look in the local neighborhood, that means range of training examples close to the value of x. Then, we will mainly focus on that local neighborhood to fit a straight line. And to actually make a prediction, we will refer to that straight line that we drew to make a prediction.

    Btw, when we say that the model will mainly focus on the local neighborhood, it doesn't mean that the other parts away from x will be disregarded, it just means that most of the weights will be put on the local neighborhood.

    In locally weighted regression, we fit theta to minimize a modified cost function where w_i is a weighting function.

    if |x_i-x| is small, w_i will be close to 1
    and if |x_i-x| is large, the w_i will be close to 0

    x will be the location where we want to make a prediction and x_i is the input x for your ith training example.

    w_i is a weighting function that has a value of between 0 and 1 that tells us how much we will pay attention to the values of (x_i, y_i) when fitting the line that we will draw

    Basically, we just added a weighting term to the cost function to determine where to pay attention. If the example x_i is far away from where we want to make a prediction, then the error term will be multiplied by a constant very close to 0. And if the x_i is near where we want to make prediction, then the error term will be multiplied by 1. This will result to the disappearance of the terms multiplied to 0 when we are summing. As such, we will only sum  the squared error terms that is near the x that we want to predict.

    If we plot the weighting terms w_i that we have, then we will notice that it looks similar with a gaussian proabability density. However it is not related to a gaussian probability density because it does not integrate to 1.

    As for choosing the width of the gaussian density that we have for w_i, we will add a bandwidth parameter tau. This will be a hyper-parameter for the algorithm. Depending on the value of tau, the shape of our bell-shaped curve could be thinner or fatter which causes us to either look in a narrower or bigger window when deciding how many nearby examples we want to use to fit the straight line.

    We have to keep in mind that our choice for the hyper-parameter tau has an effect on overfitting or underfitting of our model. If we have a tau that is too broad, we end up over-smoothing the data, and if the tau is too thin, then we will have a very jagged or eratic fit to the data

## Probabilistic Interpretation

    Why use squared error and not fourth power or other?
        - Epsilon i: is an error term that includes unmodeled effects and random noises

        - In case of the previous housing price prediction, let's assume that there's
        a true price of every house y_i which is x transpose i pllus epsilon i

        - Furthermore, we will assume that epsilon i is distributed gaussian, it would mean 0 and co-variance sigma squared.

        - The probability density of epsilon i is the gaussian density 1 over root 2 pi sigma e to the negative epsilon i squared over 2 sigma squared. And this function integrates to 1. As such, this is a probability density function.

        - Basically, we assume that the way housing prices are determined is that --- first is a true price theta transpose x; and then, some random force of nature (i.e. mood of the seller or some other factor).

    We assume that epsilon i's are IID. And IID from statistics stands for Independently and Indentically Distributed. That means that the error term for one house is independent as the error term for a different house. Although this might not be a true assumption, but it is good enough.

    When using these set of assumption, it implies that the density or the probability of y_i given x_i and theta is going to be: given x and theta, what's the probability of a particular house's price? Well, it is going to be gaussian with mean given by theta transpose x_i or theta trnaspose x, and the variance is given by sigma squared

    Notation ";" reads as parametarized by

    P(y_(i)|x_(i);theta) reads as the Probability of y_(i) given x_(i) and parametarized by theta is ..... (I need to expand this since I didn't understand it quite well)

    The random variable y given x and parametarized by theta is the distributed gaussian with that ... (I need to expand this since I didn't understand it quite well)

    L(theta) is read as the likelihood of parameters theta

    The likelihood of parameters theta is defined as the probability of the data. This is the probability of all the values of y of y1 up to ym given all the xs and given the parameters theta parametarized by theta. This is equal to the product from i equals 1 through m of p of yi given xi parametarized by theta

    Since we assume that the errors are IID, then the probability of all of the observations of all the values of y in your training set is equal to the product of the probabilities. This is because of the independence assumption we made.

    What's the difference between likelihood and probability?
        - The likelihood of the parameters is exactly the same thing as the probability of the data

        - However, we view a function of the parameters holding the data fixed, then we call that the likelihood. This means that if you think of the training set, the data as a fixed thing, then varying parameters theta, we call use the term likelihood.

        - Whereas if you view the parameters theta as a fixed and maybe varying data, then we use the term probability.

    Likelihood of the parameters
    Probability of the data

    We will use l(theta) to denote the log of the likelihood of the parameters
    and the log-likelihood is just the log of a product is equal to the sum of the logs, this is equal to m log 1 over root

    One of the well-tested letters in statistics estimating parameters is to use maximum likelihood estimation (MLE). This means we choose theta to maximize the likelihood.

    So given a dataset, how would we estimate theta?
        - We choose the value for theta depending on whatever value has the highest likelihood.
        -We choose a value of theta that will maximize the probability of data

    Classification problem to make:
        - Make an assumption about P of Y given X
        - P of Y given X parametarizes theta, and then second is to figure out the maximum likelihood estimation. Apply this framework to a different type of problem, where the value of Y is now either 0 or 1. So this will become a classification problem
        - We're going to start with binary classification because there are two classes.
        - Applying linear regression to classification problem should not be done because it is not a good idea. Linear regression is not a good algorithm for classification problem

## Logistic Regression

    In logistic regression, we might want its hypothesis output to be between 0 and 1

    This is the mathematical notation for the values for H of X or H prime, H subscript theta
    of X lies in the set from 0 to 1.

    We are going to choose the following form of hypothesis:
        - sigmoid or logistic function
        - g(z) outputs values between zero and 1

    When designing a learning algorithm, sometimes you just have to choose the form of the
    hypothesis. That means how are you gonna represent the function of h, or h subscript theta?

    There's a broader class of algorithms called generalized linear models, and both linear regression and logistic regression is part or derived using a broader set of principles.

    We will make some assumptions about the distribution of y given x parameterized by theta:
        - we will assume that the data has the following distribution --- The probability of y
          being 1 given the feature x parameterized by theta is equal to the output of our
          hypothesis
        - In case of the example of classifying whether a tumor is malignant, we want our
          learning algorithm to input the features and tell us the chance that the tumor is
          malignant. In other words, what's the chance of y is equal to 1?
        - The chance of y being 0 given x parameterized by theta is 1 minus the output of the
          hypothesis. So if the tumor has a chance of being 10 percent malignant, then there is a 90 percent chance that the tumor is benign. If we add both probabilities, it must be equal to 1.

    But we can notice that we need the y to only be either 0 or 1. Since we need 0 or 1,
    we have to take the two equations for the probability of y being 1, and y being 0, and
    compress it to a single equation.

    The resulting equation will be the probability of y given x parameterized by theta is
    equal to h of x raised to y, and multiplied by 1 minus h of x raised to 1 minus y.
        - P(y|x;theta) = (h(x)_y)(1-h(x))_1-y

    We are gonna take the log likelihood, and try to maximize l(theta)

    To summarize, if we are going the predict whether a tumor is malignant or benign, we
    will first have a training set of (x_, y_i). We will define the likelihood, define the
    log-likelihood. Then, we will need to use an algorithm such as gradient descent or gradient
    ascent, and try to find the value of theta which will maximize the log-likelihood.

    Once we have the value of theta, and a new patient comes in to the doctor's office, we can
    then take the tumor features of the new patient, and use H of theta to predict using the
    new features to determine what's the chance that the new patient's tumor is malignant or
    benign.

    The algorithm that we will use to get the theta which will maximize the log-likelihood is
    called the Batch Gradient Ascent. This function will optimize the log-likelihood instead of
    the squared cost function.

    Basically, Gradient Descent wants to climb down the gradient hill. On the other hand, Gradient
    Ascent wants to go up the gradient hill

    In batch gradient ascent, we update theta J according to the learning rate Alpha.

    In batch logistic regression, there's no local maximum. There is only the global maximum

## Newton's Method

    Gradient Ascent takes too much time to converge since it keeps taking small steps towards the global maximum.

    Luckily, we have Newton's Method which allows for less iterations before convergence. However, each iteration of this algorithm is much more expensive.

    Say you have some function f, and you want to find theta, such that f of theta is equal to
    zero. What we want is to maximize or log of likelihood of theta. And at the maximum, the
    first derivative must be 0. So we want the value where the derivative L prime of theta is equal to 0. L prime is the first derivative of theta.

    We are going to set F of theta to the L prime theta, and we are going to find the point
    where the derivative is equal to 0.

    Let's say we start of at some point in the first iteration, and we are going to find a line that is just tangent to f. To do this, we will use a straight line appromixation to f, and solve for where f touches the horizontal axis. And that point where the f touches the horizontal axis in the straight line is where our second iteration will be.

    We will repeat this process. Look at the point, draw a line that is tangent to the function of f in that point, and look for where the line touches the horizontal axis, and that place where the line touches the horizontal axis is where the new point will be. After a number of iteration, we will arrive at our goal

    We can also overshoot to where our goal is. But hopefully, it will not be much

    The range from the points of one iteration to another is denoted as uppercase Greek alphabet delta. As for the height front the horizontal point of an iteration to the f is just the f of theta_i (i.e. f(theta_0))

    In calculus, we know that the slope of the function f is the height over the run or width.
    And so we know that the derivative of delta f prime is the derivative of f at the point theta_0
        - theta_1 := theta_0 - delta
        - f'(theta_0) = (f(theta_0) / delta)
        - delta = ( (f(theta_0)) / (f'(theta_0)))
        - theta_(t+1) := theta_t - ( (f(theta_t)) / (f'(theta_t)) )
        - Let f(theta) = l'(theta)
        - theta_(t+1) := theta_t - ( (l'(theta_t)) / (l''(theta_t)) )

    Newton's method enjoys a property called quadratic convergence. This means that for example, in the first iteration, newton's method has 0.01 error (we are 0.01 away from the true minimum/maximum); after 1 iteration, the error could go to 0.0001, and in the next it could go to 0.00000001.

    Under certain assumptions in the newton's method, the function moves not too far from quadratic, the number of significant digits that we have converged, the minimum doubles on a single iteration. This is why newton's method requires fewer iterations.

    When theta is a vector:
        - theta_(t+1) := theta_t + H_(-1)(vector of derivative)
        - H_(-1) is the Hessian matrix defined as the matrix of partial derivatives

    The disadvantage of newton's method:
        - In high-dimensional problems, if there is a vector, the each step of newton's method
          is much more expensive. This is because we are either solving a linear system equations, or having to invert a big matrix.
        - Example is when we have a ten dimensional theta, we will need to invert a 10x10 matrix.
          But if we have let's say a ten-thousand dimensional theta, in that case we will need
          to invert a 10,000x10,000 matrix. In that case, it will become expensive to compute
          for each iteration.
        - If you have a small number of parameters (i.e. 15, 10, or less), using newton's method
          is great.
