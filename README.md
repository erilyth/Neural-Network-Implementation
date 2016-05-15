# Neural Network (In Python)

### Usage

* With a set of training inputs and outputs
```python
    inputs = np.array([[0, 0, 1, 1], 
                       [1, 1, 1, 1], 
                       [1, 0, 1, 1]]).T

    outputs = np.array([[0, 1, 1]])
```
* Define the shape of the network (The number of nodes in each layer). 
```python
    shape = [4, 3, 2, 1]
```
* Generate initial weights and train the network
```python
    weights = generate_network(network_shape)
    weights = train_network_main(inputs, outputs, shape, weights)
```
* Consider a test input and run the trained network, the output will be printed on the screen
```python
    test_input = np.array([[0, 0, 1, 1]]).T
    run_network(test_input, shape, weights)
```

### Methods

* sigmoid - Runs the sigmoid function on a given matrix and returns a matrix of the same dimensions
* sigmoid_derivative - Runs the sigmoid derivative function on a given matrix and returns a matrix of the same dimensions
* generate_network - Creates a randomized list of weight arrays for each layer of the network
* run_network - Gives us the predicted output for a certain set of inputs
* train_network_main - Runs the train_network method multiple times to improve our weights of the neural network
* train_network - Given a set of inputs and expected outputs, runs our network as of now and calculates the error between the predicted output and the expected output. This error is then used in the gradient descent approach and the weights of the network are adjusted accordingly such that the cost (error) is minimized. (We go down the slope so the cost is minimized, so if slope is +ve we go in the -ve direction and vice versa)

###### Tutorial

* To learn more about how neural networks work and the theory behind them, take a look at this [tutorial] by Welch Labs

[tutorial]: <https://www.youtube.com/watch?v=bxe2T-V8XRs>
