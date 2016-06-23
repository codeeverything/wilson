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
 * A scaling factor to control how far we move around the "slope" during back
 * propogation (uses the Gradient Descent algorithm), so we don't miss the
 * hopefully global minimum we're looking for. 
 * Values from 0 - 1
 */
var learningRate = 0.7;

/**
 * Number of iterations to train over
 */
var iterations = 5000;

var hiddenUnits = 3;

/**
 * Inputs - training data or real data
 */
// var inputs = new Matrix([
//     [1,1],
//     [0,0],
//     [0,1],
//     [1,0]
// ]);

/**
 * Expected output(s)
 */
// var target = new Matrix([
//     [0],
//     [0],
//     [1],
//     [1]
// ]);

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
    hidden = inputWeights.multiply(inputs);
    
    // apply the activation function
    // hidden = hidden.transform(sigmoid);  // don't do this here as we need to reference "hidden" in back prop
    
    // hidden > output
    // multiply the hidden weights by the hidden values and sum the resulting matrix (array)
    var sum = hiddenWeights.multiply(hidden.transform(sigmoid));
    
    // > output
    // return the sum and the result of sum passed through the activation function
    return {
        sum: sum, 
        val: sum.transform(sigmoid)
    };
}

/**
 * Backward propogation
 */
function backward(inputs, guess, target) {
    // error rate
    var error = target.subtract(guess.val);
    
    // output > hidden weights
    var delta = guess.sum.transform(sigmoidPrime).dot(new Matrix(error));
    delta = new Matrix(delta);
    var deltaWeights = delta.multiply(hidden.transpose().transform(sigmoid)).transform(function (val) {
        return val * learningRate;
    });
    
    var hiddenBefore = hiddenWeights;
    hiddenWeights = new Matrix(hiddenWeights.add(deltaWeights));
    
    // hidden > input weights
    delta = hiddenWeights.transpose().multiply(delta).dot(hidden.transform(sigmoidPrime));
    delta = new Matrix(delta);
    deltaWeights = delta.multiply(inputs.transpose()).transform(function (val) {
        return val * learningRate;
    });
    
    var oldInputWeights = inputWeights;
    inputWeights = new Matrix(inputWeights.add(deltaWeights));
    
    return error;
}

/**
 * Train the network on the input(s) and expected output(s)
 */
function learn(inputs, target) {
    inputs = inputs;
    target = target;
    
    inputWeights.populate(inputs.data()[0].length, hiddenUnits);
    hiddenWeights.populate(hiddenUnits, target.data()[0].length);
    // inputs = inputs.transform(sigmoid);
    // target = target.transform(sigmoid);
    
    var guesses = [];
    // log('inital');
    console.log('learning...');
    for (var i=0; i < iterations; i++) {
        // console.log('iteration', i+1);
        var guess = forward(inputs);
        guesses.push(guess.val.data());
        // log('forward');
        var error = backward(inputs, guess, target);
        // log('backward');
    }
    
    log('end');
    // console.log('guesses', guesses);
    console.log(error);
}

/**
 * Given a trained network provide some (novel) input and get an output
 */
function predict(input, expected) {
    var prediction = forward(input);
    // console.log(prediction.sum.data());
    console.log('predicted', prediction.val.data()[0][0].toFixed(3), 'expected', expected);
    return prediction.val.data()[0][0].toFixed(3);
}

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

// // train OR
// learn(new Matrix([
//     [1,1],
//     [0,0],
//     [0,1],
//     [1,0]
// ]), new Matrix([
//     [1],
//     [0],
//     [1],
//     [1]
// ]));

// // test
// predict(new Matrix([[1,1]]), 1);
// predict(new Matrix([[0,1]]), 1);
// predict(new Matrix([[1,0]]), 1);
// predict(new Matrix([[0,0]]), 0);

// // train AND
// learn(new Matrix([
//     [1,1],
//     [0,0],
//     [0,1],
//     [1,0]
// ]), new Matrix([
//     [1],
//     [0],
//     [0],
//     [0]
// ]));

// // test
// predict(new Matrix([[1,1]]), 1);
// predict(new Matrix([[0,1]]), 0);
// predict(new Matrix([[1,0]]), 0);
// predict(new Matrix([[0,0]]), 0);

// train RGB
learn(new Matrix([
    [0],
    // [20],
    // [30],
    // [40],
    // [50],
    // [60],
    // [65],
    // [70],
    // [80],
    // [85],
    // [100],
    // [200],
    // [300],
    // [400],
    // [500],
    // [600],
    // [700],
    // [800],
    // [900],
    [1],
    // [0,1,0],
    // [0,0,1],
]), new Matrix([
    [0.1],
    // [0.1],
    // [0.1],
    // [0.1],
    // [0.1],
    // [0.1],
    // [0.1],
    // [0.1],
    // [0.1],
    // [0.1],
    // [0.5],
    // [0.5],
    // [0.5],
    // [0.5],
    // [0.5],
    // [0.5],
    // [0.5],
    // [0.5],
    // [0.5],
    [0.5],
    // [0.5]
]));

// test
predict(new Matrix([[0]]), '0.1 - small');
predict(new Matrix([[1]]), '0.5 - large');
predict(new Matrix([[44]]), '0.1 - small');
predict(new Matrix([[1000]]), '0.5 - large');


// /**
//  * Letters.
//  *
//  * - Imagine these # and . represent black and white pixels.
//  */

// var a = character(
//   '.#####.' +
//   '#.....#' +
//   '#.....#' +
//   '#######' +
//   '#.....#' +
//   '#.....#' +
//   '#.....#'
// );

// var b = character(
//   '######.' +
//   '#.....#' +
//   '#.....#' +
//   '######.' +
//   '#.....#' +
//   '#.....#' +
//   '######.'
// );

// var c = character(
//   '#######' +
//   '#......' +
//   '#......' +
//   '#......' +
//   '#......' +
//   '#......' +
//   '#######'
// );

// /**
//  * Learn the letters A through C.
//  */

// learn(new Matrix([
//     a,
//     b,
//     c
// ]), new Matrix([
//     map('a'),
//     map('b'),
//     map('c')
// ]));

// /**
//  * Predict the letter C, even with a pixel off.
//  */
// var p = predict(new Matrix([
//     character(
//       '######.' +
//       '#.....#' +
//       '#.....#' +
//       '######.' +
//       '#.....#' +
//       '#.....#' +
//       '######.'
//     )
// ]), map('b'));

// var targets = [
//     0.1,
//     0.3,
//     0.5
// ];

// var rmap = ['a', 'b', 'c'];
// var bestError = 99999;
// var candidate;
// for (var i=0; i<targets.length; i++) {
//     var err = Math.sqrt((targets[i] - p) * (targets[i] - p));
//     if (err < bestError) {
//         bestError = err;
//         candidate = rmap[i];
//     }
// }

// console.log('best candidate match', candidate, 'with error of', bestError.toFixed(3), 'from expected');

// /**
//  * Turn the # into 1s and . into 0s.
//  */

// function character(string) {
//   return string
//     .trim()
//     .split('')
//     .map(integer);

//   function integer(symbol) {
//     if ('#' === symbol) return 1;
//     if ('.' === symbol) return 0;
//   }
// }

// /**
//  * Map letter to a number.
//  */

// function map(letter) {
//   if (letter === 'a') return [ 0.1 ];
//   if (letter === 'b') return [ 0.3 ];
//   if (letter === 'c') return [ 0.5 ];
//   return 0;
// }