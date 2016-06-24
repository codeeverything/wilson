/**
 * Wilson - A simple JS neural network
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
 * Inspired and heavily influenced by:
 * https://github.com/stevenmiller888/mind
 * http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
 * https://www.youtube.com/watch?v=bxe2T-V8XRs
 * https://github.com/harthur/brain/blob/master/lib/neuralnetwork.js
 * http://iamtrask.github.io/2015/07/12/basic-python-network/
 * http://iamtrask.github.io/2015/07/27/python-network-part2/
 * 
 * @author Mike Timms <mike@codeeverything.com>
 */

/**
 * Dependencies
 */
var linearAlgebra = require('linear-algebra')(),    // initialise it 
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
 * Wilson
 */

// export Wilson
module.exports = Wilson;

function Wilson(opts) {
    // config options
    opts = opts || {};
    
    // hyper-paramters
    var hiddenNodes;    // number of hidden neurons
    var iterations;  // number of iterations
    var learningRate;
    
    // confgigure the network
    config();
    
    // weights and values
    
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
     * Sigmoid to derivitive
     */
    function sigmoidOutputToDerivitive(a) {
        return a * (1-a);
    }
    
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
    
    function config() {
        hiddenNodes = opts.hiddenNodes || 3;    // number of hidden neurons
        iterations = opts.iterations || 10000;  // number of iterations
        learningRate = opts.learningRate || 0.1;
    }
    
    /**
     * Forward propogation
     */
    function forward(inputs) {
        // input > hidden
        // multiply the input weights by the inputs
        hidden = inputs.dot(inputWeights).sigmoid();
        
        // hidden > output
        // multiply the hidden weights by the hidden values and sum the resulting matrix (array)
        var sum = hidden.dot(hiddenWeights).sigmoid();
        
        // > output
        return sum;
    }
    
    /**
     * Backward propogation
     * 
     * Uses Stochastic Gradient Descent to optimise
     */
    function backward(inputs, guess, target) {
        // output layer error
        var error = guess.minus(target);
        
        var outputDelta = error.mul(guess.map(sigmoidOutputToDerivitive));
        
        // hidden layer error
        var hiddenLayerError = outputDelta.dot(hiddenWeights.trans());
        var hiddenLayerDelta = hiddenLayerError.mul(hidden.map(sigmoidOutputToDerivitive));
        
        // adjust hidden > output weights
        hiddenWeights = hiddenWeights.minus(hidden.trans().dot(outputDelta).map(function (val) {
            return val * learningRate;
        }));
        
        // adjust input > hidden weights
        inputWeights = inputWeights.minus(inputs.trans().dot(hiddenLayerDelta).map(function (val) {
            return val * learningRate;
        }));
        
        // return the error
        return error
    }
    
    // expose methods and properties
    return {
        learn: function (inputs, target) {
            // first configure the network
            config();
            
            // learn the outputs from the inputs
            inputs = new Matrix(inputs);
            target = new Matrix(target);
            
            // initialize weights
            inputWeights = inputWeights.populate(inputs.toArray()[0].length, hiddenNodes);
            hiddenWeights = hiddenWeights.populate(hiddenNodes, target.toArray()[0].length);
            
            // learn yourself something
            for (var i=0; i < iterations; i++) {
                var error = backward(inputs, forward(inputs), target);
                
                // output error margin every 1000 iterations
                if (i % 1000 == 0) {
                    var err = (function(err) {
                        return Math.abs(error.trans().toArray()[0].reduce(function (a, b) {
                            return a + b;
                        }) / error.trans().toArray()[0].length);
                    })(error);
                    
                    console.log('Error after ', i, 'iterations', err);
                }
            }
        },
        predict: function (input, expected) {
            // first configure the network
            config();
            
            // predict the output from the input
            var prediction = forward(new Matrix(input));
            console.log('predicted', prediction.toArray()[0][0].toFixed(2), 'expected', expected);
            return prediction.toArray()[0][0].toFixed(2);
        },
        configure: function (conf) {
            // update any current config with values passed
            for (var op in conf) {
                opts[op] = conf[op];
            }
        },
        save: function () {
            // return a JSON string of the weights/parameters
            return JSON.stringify({
                hyperParams: {
                    hiddenNodes: opts.hiddenNodes,
                    // iterations: opts.iterations,
                    // learningRate: opts.learningRate
                },
                weights: [
                    inputWeights.toArray(),
                    hiddenWeights.toArray()
                ]
            });
        },
        load: function (config) {
            // load the given network config (weights and hyper-parameters)
            config = JSON.parse(config);
            
            // setup
            opts = {
                hiddenNodes: config.hyperParams.hiddenNodes,
                iterations: config.hyperParams.iterations,
                learningRate: config.hyperParams.learningRate
            };
            
            // load up the values
            config();
            
            inputWeights = config.weights[0];
            hiddenWeights = config.weights[1];
        }
    }
}