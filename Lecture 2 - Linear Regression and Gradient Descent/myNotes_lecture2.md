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