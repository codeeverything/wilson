# Wilson

IBM Watson's very distant cousin.

Wilson is an experiment in artificial neural networks. It's a work in progress, but has been tested to work on simple problems like the XOR truth table and IRIS dataset.

Wilson is implemented in NodeJS, but there's no reason you couldn't fairly easily make it work in the browser, or port to another language (indeed, Wilson is largely based off of other work written in Python).

### Tests/examples

Run ```$ node test.js```, and ```$ node testLetter.js```

The former runs the following tests: XOR, RGB and IRIS

The latter runs the example test from the [Mind neural network](https://github.com/stevenmiller888/mind) by Steven Miller, which was a huge inspiration in writing Wilson :)

### Notes

- Wilson is kind of focused on classification at the moment and even binary output will result in two output nodes with a confidence score each. I.e. Wilson doesn't handle regression problems (those where you want a single value as a result), at the moment. I'm looking to add that as an option.

- Wilson also always uses [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation on the output layer to give a "probability" score for each node, where the total of all scores is 1.

### Learning the XOR truth table

We can learn the XOR truth table as follows:

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