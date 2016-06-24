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
            res[i][j] = Math.random() - 1;
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

function sigmoidOutputToDerivitive(a) {
    return a * (1-a);
}

/**
 * A scaling factor to control how far we move around the "slope" during back
 * propogation (uses the Gradient Descent algorithm), so we don't miss the
 * hopefully global minimum we're looking for. 
 * Values from 0 - 1
 */
var learningRate = 0.1;

/**
 * Number of iterations to train over
 */
var iterations = 10000;

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
    hidden = inputs.dot(inputWeights).sigmoid();
    
    // apply the activation function
    // hidden = hidden.transform(sigmoid);  // don't do this here as we need to reference "hidden" in back prop
    
    // hidden > output
    // multiply the hidden weights by the hidden values and sum the resulting matrix (array)
    var sum = hidden.dot(hiddenWeights).sigmoid();
    
    // > output
    return sum;
}

/**
 * Backward propogation
 */
function backward(inputs, guess, target) {
    // output layer error
    var error = guess.minus(target);    //layer_2_error = layer_2 - y
    
    var outputDelta = error.mul(guess.map(sigmoidOutputToDerivitive)); //layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)
    
    // hidden layer error
    var hiddenLayerError = outputDelta.dot(hiddenWeights.trans());    //layer_1_error = layer_2_delta.dot(synapse_1.T)
    var hiddenLayerDelta = hiddenLayerError.mul(hidden.map(sigmoidOutputToDerivitive));    //layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    
    // adjust hidden > output weights
    hiddenWeights = hiddenWeights.minus(hidden.trans().dot(outputDelta).map(function (val) {
        return val * learningRate;
    }));   //synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
    
    // adjust input > hidden weights
    inputWeights = inputWeights.minus(inputs.trans().dot(hiddenLayerDelta).map(function (val) {
        return val * learningRate;
    }));    //synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
    
    // return the error
    return error
}

/**
 * Train the network on the input(s) and expected output(s)
 */
function learn(inputs, target) {
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
        if (i % 1000 == 0) {
            // console.log(error);
            var err = (function(err) {
                return Math.abs(error.trans().toArray()[0].reduce(function (a, b) {
                    return a + b;
                }) / error.trans().toArray()[0].length);
            })(error);
            
            console.log('Error after ', i, 'iterations', err);
            
            // if (err < 0.005) {
            //     break;
            // }
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
    // console.log(prediction);
    console.log('predicted', prediction.toArray()[0][0].toFixed(2), 'expected', expected);
    return prediction.toArray()[0][0].toFixed(2);
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

// train OR
learn(new Matrix([
    [1,1],
    [0,0],
    [0,1],
    [1,0]
]), new Matrix([
    [1],
    [0],
    [1],
    [1]
]));

// test
predict(new Matrix([[1,1]]), 1);
predict(new Matrix([[0,1]]), 1);
predict(new Matrix([[1,0]]), 1);
predict(new Matrix([[0,0]]), 0);

// train OR
learn(new Matrix([
    [1.1,1.1],
    [0,0],
    [0,1.1]
]), new Matrix([
    [0.5],
    [0.1],
    [0.3]
]));

// test
predict(new Matrix([[1.1,1.1]]), 0.5);
predict(new Matrix([[0,1.1]]), 0.3);
predict(new Matrix([[0,0]]), 0.1);
predict(new Matrix([[1,0]]), 0.1);

// train IRIS
learn(new Matrix([
    [5.1,3.5,1.4,0.2  ],
    [4.9,3,1.4,0.2  ],
    [4.7,3.2,1.3,0.2  ],
    [4.6,3.1,1.5,0.2  ],
    [5,3.6,1.4,0.2  ],
    [5.4,3.9,1.7,0.4  ],
    [4.6,3.4,1.4,0.3  ],
    [5,3.4,1.5,0.2  ],
    [4.4,2.9,1.4,0.2  ],
    [4.9,3.1,1.5,0.1  ],
    [5.4,3.7,1.5,0.2  ],
    [4.8,3.4,1.6,0.2  ],
    [4.8,3,1.4,0.1  ],
    [4.3,3,1.1,0.1  ],
    [5.8,4,1.2,0.2  ],
    [5.7,4.4,1.5,0.4  ],
    [5.4,3.9,1.3,0.4  ],
    [5.1,3.5,1.4,0.3  ],
    [5.7,3.8,1.7,0.3  ],
    [5.1,3.8,1.5,0.3  ],
    [5.4,3.4,1.7,0.2  ],
    [5.1,3.7,1.5,0.4  ],
    [4.6,3.6,1,0.2  ],
    [5.1,3.3,1.7,0.5  ],
    [4.8,3.4,1.9,0.2  ],
    [5,3,1.6,0.2  ],
    [5,3.4,1.6,0.4  ],
    [5.2,3.5,1.5,0.2  ],
    [5.2,3.4,1.4,0.2  ],
    [4.7,3.2,1.6,0.2  ],
    [4.8,3.1,1.6,0.2  ],
    [5.4,3.4,1.5,0.4  ],
    [5.2,4.1,1.5,0.1  ],
    [5.5,4.2,1.4,0.2  ],
    [4.9,3.1,1.5,0.1  ],
    [5,3.2,1.2,0.2  ],
    [5.5,3.5,1.3,0.2  ],
    [4.9,3.1,1.5,0.1  ],
    [4.4,3,1.3,0.2  ],
    [5.1,3.4,1.5,0.2  ],
    [5,3.5,1.3,0.3  ],
    [4.5,2.3,1.3,0.3  ],
    [4.4,3.2,1.3,0.2  ],
    [5,3.5,1.6,0.6  ],
    [5.1,3.8,1.9,0.4  ],
    [4.8,3,1.4,0.3  ],
    [5.1,3.8,1.6,0.2  ],
    [4.6,3.2,1.4,0.2  ],
    [5.3,3.7,1.5,0.2  ],
    [5,3.3,1.4,0.2  ],
    [7,3.2,4.7,1.4  ],
    [6.4,3.2,4.5,1.5  ],
    [6.9,3.1,4.9,1.5  ],
    [5.5,2.3,4,1.3  ],
    [6.5,2.8,4.6,1.5  ],
    [5.7,2.8,4.5,1.3  ],
    [6.3,3.3,4.7,1.6  ],
    [4.9,2.4,3.3,1  ],
    [6.6,2.9,4.6,1.3  ],
    [5.2,2.7,3.9,1.4  ],
    [5,2,3.5,1  ],
    [5.9,3,4.2,1.5  ],
    [6,2.2,4,1  ],
    [6.1,2.9,4.7,1.4  ],
    [5.6,2.9,3.6,1.3  ],
    [6.7,3.1,4.4,1.4  ],
    [5.6,3,4.5,1.5  ],
    [5.8,2.7,4.1,1  ],
    [6.2,2.2,4.5,1.5  ],
    [5.6,2.5,3.9,1.1  ],
    [5.9,3.2,4.8,1.8  ],
    [6.1,2.8,4,1.3  ],
    [6.3,2.5,4.9,1.5  ],
    [6.1,2.8,4.7,1.2  ],
    [6.4,2.9,4.3,1.3  ],
    [6.6,3,4.4,1.4  ],
    [6.8,2.8,4.8,1.4  ],
    [6.7,3,5,1.7  ],
    [6,2.9,4.5,1.5  ],
    [5.7,2.6,3.5,1  ],
    [5.5,2.4,3.8,1.1  ],
    [5.5,2.4,3.7,1  ],
    [5.8,2.7,3.9,1.2  ],
    [6,2.7,5.1,1.6  ],
    [5.4,3,4.5,1.5  ],
    [6,3.4,4.5,1.6  ],
    [6.7,3.1,4.7,1.5  ],
    [6.3,2.3,4.4,1.3  ],
    [5.6,3,4.1,1.3  ],
    [5.5,2.5,4,1.3  ],
    [5.5,2.6,4.4,1.2  ],
    [6.1,3,4.6,1.4  ],
    [5.8,2.6,4,1.2  ],
    [5,2.3,3.3,1  ],
    [5.6,2.7,4.2,1.3  ],
    [5.7,3,4.2,1.2  ],
    [5.7,2.9,4.2,1.3  ],
    [6.2,2.9,4.3,1.3  ],
    [5.1,2.5,3,1.1  ],
    [5.7,2.8,4.1,1.3  ],
    [6.3,3.3,6,2.5  ],
    [5.8,2.7,5.1,1.9  ],
    [7.1,3,5.9,2.1  ],
    [6.3,2.9,5.6,1.8  ],
    [6.5,3,5.8,2.2  ],
    [7.6,3,6.6,2.1  ],
    [4.9,2.5,4.5,1.7  ],
    [7.3,2.9,6.3,1.8  ],
    [6.7,2.5,5.8,1.8  ],
    [7.2,3.6,6.1,2.5  ],
    [6.5,3.2,5.1,2  ],
    [6.4,2.7,5.3,1.9  ],
    [6.8,3,5.5,2.1  ],
    [5.7,2.5,5,2  ],
    [5.8,2.8,5.1,2.4  ],
    [6.4,3.2,5.3,2.3  ],
    [6.5,3,5.5,1.8  ],
    [7.7,3.8,6.7,2.2  ],
    [7.7,2.6,6.9,2.3  ],
    [6,2.2,5,1.5  ],
    [6.9,3.2,5.7,2.3  ],
    [5.6,2.8,4.9,2  ],
    [7.7,2.8,6.7,2  ],
    [6.3,2.7,4.9,1.8  ],
    [6.7,3.3,5.7,2.1  ],
    [7.2,3.2,6,1.8  ],
    [6.2,2.8,4.8,1.8  ],
    [6.1,3,4.9,1.8  ],
    [6.4,2.8,5.6,2.1  ],
    [7.2,3,5.8,1.6  ],
    [7.4,2.8,6.1,1.9  ],
    [7.9,3.8,6.4,2  ],
    [6.4,2.8,5.6,2.2  ],
    [6.3,2.8,5.1,1.5  ],
    [6.1,2.6,5.6,1.4  ],
    [7.7,3,6.1,2.3  ],
    [6.3,3.4,5.6,2.4  ],
    [6.4,3.1,5.5,1.8  ],
    [6,3,4.8,1.8  ],
    [6.9,3.1,5.4,2.1  ],
    [6.7,3.1,5.6,2.4  ],
    [6.9,3.1,5.1,2.3  ],
    [5.8,2.7,5.1,1.9  ],
    [6.8,3.2,5.9,2.3  ],
    [6.7,3.3,5.7,2.5  ],
    [6.7,3,5.2,2.3  ],
    [6.3,2.5,5,1.9  ],
    [6.5,3,5.2,2  ],
    [6.2,3.4,5.4,2.3  ],
    // [5.9,3,5.1,1.8  ]
]), new Matrix([
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-versicolor",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    "Iris-virginica",
    // "Iris-virginica"
].map(flowerToDec)));

function flowerToDec(flower) {
    if (flower == "Iris-setosa") return [0.1];
    if (flower == "Iris-versicolor") return [0.3];
    if (flower == "Iris-virginica") return [0.5];
    return 0;
}

function decToFlower(dec) {
    if (dec == 0.1) return "Iris-setosa";
    if (dec == 0.3) return "Iris-versicolor";
    if (dec == 0.5) return "Iris-virginica";
    return "unsure";
}
// trained 
var p = predict(new Matrix([[5.1,3.5,1.4,0.2]]), 0.1);
console.log(decToFlower(p));
// unknown
p = predict(new Matrix([[5.9,3,5.1,1.8]]), 0.5);
console.log(decToFlower(p));

/**
 * Letters.
 *
 * - Imagine these # and . represent black and white pixels.
 */

var a = character(
  '.#####.' +
  '#.....#' +
  '#.....#' +
  '#######' +
  '#.....#' +
  '#.....#' +
  '#.....#'
);

var b = character(
  '######.' +
  '#.....#' +
  '#.....#' +
  '######.' +
  '#.....#' +
  '#.....#' +
  '######.'
);

var c = character(
  '#######' +
  '#......' +
  '#......' +
  '#......' +
  '#......' +
  '#......' +
  '#######'
);

/**
 * Learn the letters A through C.
 */

learningRate = 10;
learn(new Matrix([
    a,b,c    
]), new Matrix([
    map('a'),
    map('b'),
    map('c'),
]));

/**
 * Predict the letter C, even with a pixel off.
 */

predict(new Matrix([
    character(
        '.######' +
        '#......' +
        '#......' +
        '#......' +
        '#......' +
        '##.....' +
        '.......'
    )
]), 0.5);

/**
 * Turn the # into 1s and . into 0s.
 */

function character(string) {
  return string
    .trim()
    .split('')
    .map(integer);

  function integer(symbol) {
    if ('#' === symbol) return 1;
    if ('.' === symbol) return 0;
  }
}

/**
 * Map letter to a number.
 */

function map(letter) {
  if (letter === 'a') return [ 0.1 ];
  if (letter === 'b') return [ 0.3 ];
  if (letter === 'c') return [ 0.5 ];
  return 0;
}