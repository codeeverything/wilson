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

// var Matrix = require('./matrix.js');
var linearAlgebra = require('linear-algebra')(),     // initialise it 
Vector = linearAlgebra.Vector,
Matrix = linearAlgebra.Matrix;

Matrix.prototype.populate = function (x, y) {
    function sample() {
        return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
    
    var res = [];
    for (var i=0; i < x; i++) {
        res[i] = [];
        for (var j=0; j < y; j++) {
            res[i][j] = sample();
        }
    }
    
    return new Matrix(res);
}

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
 * A scaling factor to control how far we move around the "slope" during back
 * propogation (uses the Gradient Descent algorithm), so we don't miss the
 * hopefully global minimum we're looking for. 
 * Values from 0 - 1
 */
var learningRate = 1;

/**
 * Number of iterations to train over
 */
var iterations = 100;

var hiddenUnits = 3;

/**
 * Hidden layer (just one for now), values
 */
var hidden = new Matrix([]);

/**
 * Weights between input(s) > hidden layer (1)
 */
var inputWeights = new Matrix([]);

/**
 * Weights between hidden layer (1) > output(s)
 */
var hiddenWeights = new Matrix([]);

// console.log(inputWeights.data(), hiddenWeights.data());

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
    hidden = inputs.dot(inputWeights);
    
    // apply the activation function
    // hidden = hidden.transform(sigmoid);  // don't do this here as we need to reference "hidden" in back prop
    
    // hidden > output
    // multiply the hidden weights by the hidden values and sum the resulting matrix (array)
    var sum = hidden.sigmoid().dot(hiddenWeights);
    
    // > output
    return sum;
}

/**
 * Backward propogation
 */
function backward(inputs, guess, target) {
    var error = target.minus(guess).map(function (val) {
        return val * -1;
    }); // -(y-yHat)
    
    // delta3 = delta for output to hidden weights
    var delta3 = error.mul(guess.map(sigmoidPrime)); //.mul(error);   //np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
    var dJdW2 = hidden.sigmoid().trans().dot(delta3).map(function (val) {
        return val * learningRate;
    }); //np.dot(self.a2.T, delta3)
    console.log('delta3', delta3);
    console.log('dJdW2', dJdW2);
    console.log('hidden weights before', hiddenWeights);
    var hiddenWeightsBefore = hiddenWeights.clone();
    hiddenWeights = hiddenWeights.minus(dJdW2);
    console.log('hidden weights after', hiddenWeights);
    
    // delta2 = delta for hidden to input
    console.log('delta2 ex', delta3.dot(hiddenWeightsBefore.trans()));
    var delta2 = delta3.dot(hiddenWeightsBefore.trans()).dot(hidden.map(sigmoidPrime));   //np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
    var dJdW1 = inputs.trans().dot(delta2).map(function (val) {
        return val * learningRate;
    }); //np.dot(X.T, delta2);
    console.log('dJdW1', dJdW1);
    console.log('input weights before', inputWeights);
    inputWeights = inputWeights.minus(dJdW1);
    console.log('input weights after', inputWeights);
     
    return error.toArray();
}

/**
 * Train the network on the input(s) and expected output(s)
 */
function learn(inputs, target) {
    // inputs = inputs;
    // target = target;
    
    inputWeights = inputWeights.populate(inputs.toArray()[0].length, hiddenUnits);
    hiddenWeights = hiddenWeights.populate(hiddenUnits, target.toArray()[0].length);
    
    var guesses = [];
    var errors = [];
    // log('inital');
    console.log('learning...');
    for (var i=0; i < iterations; i++) {
        // console.log('iteration', i+1);
        var guess = forward(inputs);
        guesses.push(guess.toArray());
        // log('forward');
        var error = backward(inputs, guess, target);
        var mse = 0;
        for (var e in error) {
            mse += Math.pow(error[e], 2);
        }
        errors.push(mse);
        if (mse <= 0.005) {
            console.log(errors);
            console.log('error threshold reached at iteration', i);
            return;
        }
        // log('backward');
    }
    
    log('end');
    // console.log('guesses', guesses);
    console.log(errors);
}

/**
 * Given a trained network provide some (novel) input and get an output
 */
function predict(input, expected) {
    var prediction = forward(input);
    // console.log(prediction.sum.data());
    console.log('predicted', prediction.toArray()[0][0].toFixed(3), 'expected', expected);
    return prediction.toArray()[0][0].toFixed(3);
}
// train hours and score
learn(new Matrix([
    [3,5],
    [5,1],
    [10,2],
]), new Matrix([
    [0.75],
    [0.82],
    [0.93],
]));

// test
predict(new Matrix([[3,5]]), 75);
predict(new Matrix([[5,1]]), 82);
predict(new Matrix([[10,2]]), 93);

// // train XOR
// learn(new Matrix([
//     [1,1],
//     [0,0],
//     [0,1],
//     [1,0]
// ]), new Matrix([
//     [0],
//     [0],
//     [1],
//     [1]
// ]));

// // test
// predict(new Matrix([[1,1]]), 0);
// predict(new Matrix([[0,1]]), 1);
// predict(new Matrix([[1,0]]), 1);
// predict(new Matrix([[0,0]]), 0);
