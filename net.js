/**
 * Wilson - A simple JS neural network library
 * 
 * Supports:
 * Training data with labels
 * Multiple hidden layers
 * Hidden layer size definition
 * Multiple outputs
 * Import/export of weights
 * Learning
 * Prediction
 * Hyperparameter config
 * Browser and NodeJS environments
 * 
 */

var Matrix = require('./matrix.js');



/**
 * Sigmoid "squashing" function
 */
function sigmoid(t){
   return 1 / (1 + Math.exp(-t));
}

/**
 * Derivitive of Sigmoid function
 */
function sigmoidPrime(t) {
	return (Math.exp(-t) / Math.pow(1 + Math.exp(-t), 2));
}

/**
 * Inputs - training data or real data
 */
var inputs = new Matrix([
    [1,1]
]);

/**
 * Expected output(s)
 */
var target = new Matrix([
    [1]
]);

/**
 * Hidden layer (just one for now), values
 */
var hidden = new Matrix([]);

/**
 * Actual ouput(s) from the network
 */
var outputs;

/**
 * Weights between input(s) > hidden layer (1)
 */
var inputWeights = new Matrix([
    [0.8, 0.4, 0.3],
    [0.2, 0.9, 0.5]
]);

/**
 * Weights between hidden layer (1) > output(s)
 */
var hiddenWeights = new Matrix([
    [0.3, 0.5, 0.9]
]);

/**
 * A scaling factor to control how far we move around the "slope" during back
 * propogation (uses the Gradient Descent algorithm), so we don't miss the
 * hopefully global minimum we're looking for. 
 * Values from 0 - 1
 */
var learningRate = 0.5;

/**
 * Number of iterations to train over
 */
var iterations = 200;

/**
 * Helper function to log the state of the network at a given point
 */
function log(id) {
    return;
    console.log('STATE AT ' + id);
    console.log('inputs', JSON.stringify(inputs.data(), null, 4));
    console.log('input > hidden weights', JSON.stringify(inputWeights.data(), null, 4));
    console.log('hidden values', JSON.stringify(hidden.data(), null, 4));
    console.log('hidden > output weights', JSON.stringify(hiddenWeights.data(), null, 4));
}

/**
 * Forward propogation
 */
function forward(inputs) {
    // input > hidden
    // multiply the input weights by the inputs
    hidden = inputWeights.multiply(inputs);
    
    // apply the activation function
    hidden = hidden.transform(sigmoid);
    
    // hidden > output
    // multiply the hidden weights by the hidden values and sum the resulting matrix (array)
    var sum = hiddenWeights.multiply(hidden).data()[0].reduce(function(a, b){return a+b;});
    
    // > output
    // return the sum and the result of sum passed through the activation function
    return {
        sum: sum, 
        val: sigmoid(sum)
    };
}

/**
 * Backward propogation
 */
function backward(inputs, guess) {
    var foo = new Matrix([[guess.val]]);
    
    var error = target.subtract(foo).reduce(function(a, b){return a+b;});
    
    var delta = sigmoidPrime(guess.sum) * error;
    delta = parseFloat(delta.toFixed(3));
    
    // hidden to output weights
    var hiddenBefore = hiddenWeights;
    var deltaWeights = hidden.transform(function (val) {
        return ((delta / val) * learningRate);
    });
    
    hiddenWeights = new Matrix(hiddenWeights.add(deltaWeights));
    
    // input to hidden weights
    var deltaHiddenSum = hiddenBefore.transform(function (val) {
       return parseFloat((delta / val).toFixed(3)); 
    });
    
    var sum = inputWeights.multiply(inputs);
    sum = sum.transform(sigmoidPrime);
    deltaHiddenSum = deltaHiddenSum.multiply(sum);
    deltaHiddenSum = deltaHiddenSum.transform(function (val) {
       return parseFloat((val * learningRate).toFixed(3)); 
    });
 
    var deltaWeights = deltaHiddenSum.multiply(inputs.transpose());
    var oldInputWeights = inputWeights;
    inputWeights = new Matrix(inputWeights.add(deltaWeights));
}

/**
 * Train the network on the input(s) and expected output(s)
 */
function learn() {
    var guesses = [];
    log('inital');
    for (var i=0; i < iterations; i++) {
        console.log('iteration', i+1);
        var guess = forward(inputs);
        guesses.push(guess);
        log('forward');
        backward(inputs, guess);
        log('backward');
    }
    
    console.log('guesses', guesses);
}

/**
 * Given a trained network provide some (novel) input and get an output
 */
function predict(input) {
        console.log('predicted', forward(input).val);
}

// train
learn();

// test
predict(new Matrix([[1,0]]));