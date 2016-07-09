# Wilson

IBM Watson's very distant cousin.

Wilson is an experiment in artificial neural networks. It's a work in progress, but has been tested to work on simple problems like the XOR truth table and IRIS dataset.

Wilson is implemented in NodeJS, but there's no reason you couldn't fairly easily make it work in the browser, or port to another language (indeed, Wilson is largely based off of other work written in Python).

##### New to Artificial Neural Networks?

If you're new to ANNs then I can recommend the following two part blog series from Steven Miller and the video series by Stephen Welch as primers on the topic:

- http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
- https://www.youtube.com/watch?v=bxe2T-V8XRs

### Regression and Classification

Wilson can work with both regression and classification problems. 

#### Regression

Regression problems have a single output which is a continuous value, which will have been scaled by the activation function.

#### Classification

Classification problems typically have 3 or more outputs, the value of which is the confidence of that output being correct.

> ##### Two outputs
>
> You may have noticed I missed the case of 2 outputs above.
>
> Binary output can be solved with either regression or classification. In the former values 0 - 0.5 would be considered "off" and values > 0.5 "on" (assuming a sigmoid activation function). In the latter you'd have a confidence measure for each output between 0 and 1.

#### Examples

An example (taken from Stephen Welch's video series), is predicting test score from hours studied and hours slept.

You can also approach the XOR problem as either a regression or classification problem. As a regression problem you'd expect a value output that is close to the expected 0 or 1 value, and as a classification problem you'd expect a high confidence on either the 0 or 1 output.

### Tests/examples

Run ```$ node test.js```, and ```$ node testLetter.js```

The former runs the following tests: XOR, RGB and IRIS

The latter runs the example test from the [Mind neural network](https://github.com/stevenmiller888/mind) by Steven Miller, which was a huge inspiration in writing Wilson :)

Test outputs have the following structure:

```
************************************************************
Running "XOR" test...
    Learning...
    Inputs: [[0,0],[0,1],[1,0],[1,1]]...
    Targets: [0,1,1,0]...
Error after  0 iterations 0.5492247103934262
Error after  1000 iterations 0.4994459891975506
Error after  2000 iterations 0.48036100264513476
Error after  3000 iterations 0.3318690742300831
Error after  4000 iterations 0.210235976094908
Error after  5000 iterations 0.10030507993581103
Error after  6000 iterations 0.05475651370061447
Error after  7000 iterations 0.035935203398354446
Error after  8000 iterations 0.026303441483969545
Error after  9000 iterations 0.020580266156410666
Predicting...
    Input: [[0,0]]...
    Expected: 0
    Output: 0 (93.44% confidence)
Predicting...
    Input: [[0,1]]...
    Expected: 1
    Output: 1 (90.96% confidence)
Predicting...
    Input: [[1,0]]...
    Expected: 1
    Output: 1 (88.77% confidence)
Predicting...
    Input: [[1,1]]...
    Expected: 0
    Output: 0 (90.75% confidence)
```

Hopefully this is fairly self explanatory, but in breif:

- Wilson "learns" from the training data (inputs and labels) supplied as to the test
- We should the first 5 rows of input data and target output (labels)
- Next we see a breakdown of the error (difference between the calculated output and target across all training examples), after each 1,000 iterations
- Then we see the output of various predictions made after training is complete
  - We see the input given, the output expected and the actual output (along with a confidence rating)

### Using Wilson

The basic use case for Wilson, as it currently stands, is for classification of data. As with any artificial neural network (ANN), you first provide Wilson with some training data. This consists of some set of input "features" and expected output "labels".

The input features describe properties of whatever you're trying to classify. For example, one of the tests (the last in test.js), uses a dataset which describes various properties of 3 species of Iris flowers (such as sepal length and width), along with a label for which species each row of input represents.

You pass the input and expected output to Wilson in the learn() method. Wilson then iterates X times (default is 10,000), and adjusts itself (this is glossing over the detail), such that the output it produces approaches the expected output for all training examples.

With the network trained you can then call predict() with some novel input (though still for the same sort of data, in this case Iris flowers), and Wilson will predict which "class" (flower), this data represents - with some degree of confidence.

#### Example: Learning the XOR truth table

A basic ANN example is usually to learn the XOR truth table, which we can do with Wilson as follows:

```
// get wilson
var wilson = require('./wilson.js')();

// learn XOR truth table
wilson.learn([
    [1,1],
    [0,0],
    [0,1],
    [1,0]
], [
    0,
    0,
    1,
    1
]);

// test
wilson.predict([[1,1]], 0); // 0
wilson.predict([[0,1]], 1); // 1
wilson.predict([[1,0]], 1); // 1
wilson.predict([[0,0]], 0); // 0
```

### Configuring Wilson's Hyper-parameters

Wilson allows you to configure the following hyper-parameters via the ```.configure({optName: optValue, ...)``` method:

- Number of hidden nodes
- Number of iterations
- Learning rate
- Activation function: Can be "sigmoid", "tanh" or a callable
- Derivative of activation function: Can be "sigmoidPrime", "tanhPrime" or a callable

## License

MIT

## Future work

- Normalise inputs, Encode "raw" input to numeric values/representations
- Multiple hidden layers
- Selection of most likely output(s) - i.e. predict will return the K outputs with the highest values (which may be probabilities)
- Biases
- Drop out
- Momentum, Annealing of learning rate 
- Better Tests
- Back prop algorithm: SGD, BFGS, Mini-Batch GD...?
- Regression problems - i.e. single value output, for example test score based on hours sleep and study

## References

- https://github.com/stevenmiller888/mind
- http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
- https://www.youtube.com/watch?v=bxe2T-V8XRs
- https://github.com/harthur/brain/blob/master/lib/neuralnetwork.js
- http://iamtrask.github.io/2015/07/12/basic-python-network/
- http://iamtrask.github.io/2015/07/27/python-network-part2/
- http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
- https://www.mathsisfun.com/calculus/derivatives-introduction.html
- http://sebastianruder.com/optimizing-gradient-descent/
- https://www.youtube.com/watch?v=-zT1Zi_ukSk&list=WL&index=49
- https://visualstudiomagazine.com/articles/2014/01/01/how-to-standardize-data-for-neural-networks.aspx
- http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
- https://www.willamette.edu/~gorr/classes/cs449/momrate.html
- https://www.youtube.com/watch?v=v8be6yPsl2s