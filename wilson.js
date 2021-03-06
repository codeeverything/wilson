/**
 * Wilson - A simple JS neural network
 * IBM Watson's very distant cousin
 * 
 * Supports:
 * Training data with labels - PARTIAL, correctly produces outputLayer and labels, but fails to learn the Isis dataset (smaller sets work?)
 * Hidden layer size definition - DONE
 * Multiple outputs - DONE
 * Import/export of weights - DONE
 * Learning - DONE
 * Prediction - DONE
 * Hyperparameter config - DONE 
 * Browser and NodeJS environments
 * HTAN/custom activation function - DONE
 * Softmax for probability of each output - DONE
 * 
 * Future work:
 * Normalise inputs
 * Encode "raw" input to numeric values/representations
 * Multiple hidden layers - Restructure: layers[], weights[] vs specific vars?
 * Selection of most likely output(s) - i.e. predict will return the K outputs with the highest values (which may be probabilities)
 * Biases?
 * Drop out?
 * Momentum?
 * Activation function selection per layer?
 * Annealing of learning rate
 * Tests
 * Back prop algorithm: SGD, BFGS, Mini-Batch GD...?
 * Learn best network topology?
 * 
 * Inspired and heavily influenced by:
 * https://github.com/stevenmiller888/mind
 * http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
 * https://www.youtube.com/watch?v=bxe2T-V8XRs
 * https://github.com/harthur/brain/blob/master/lib/neuralnetwork.js
 * http://iamtrask.github.io/2015/07/12/basic-python-network/
 * http://iamtrask.github.io/2015/07/27/python-network-part2/
 * http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
 * https://www.mathsisfun.com/calculus/derivatives-introduction.html
 * http://sebastianruder.com/optimizing-gradient-descent/
 * https://www.youtube.com/watch?v=-zT1Zi_ukSk&list=WL&index=49
 * https://visualstudiomagazine.com/articles/2014/01/01/how-to-standardize-data-for-neural-networks.aspx
 * http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
 * https://www.willamette.edu/~gorr/classes/cs449/momrate.html
 * https://www.youtube.com/watch?v=v8be6yPsl2s
 * 
 * @author Mike Timms <mike@codeeverything.com>
 */

/**
 * Dependencies
 * 
 * Linear Algebra - for Matrix manipulation: https://www.npmjs.com/package/linear-algebra
 */
var linearAlgebra = require('linear-algebra')(),    // initialise it 
Matrix = linearAlgebra.Matrix;

/**
 * Populates a Matrix of dimensions (x, y) with random values from the Guassian distribution 
 * 
 * @see https://github.com/stevenmiller888/sample 
 * @param int x - Number of columns
 * @param int y - Number of rows
 * @return Matrix
 */
Matrix.prototype.populate = function (x, y) {
    /**
     * @see: http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
     * 
     * According to Hugo Larochelle, Glorot & Bengio (2010), initialize the weights uniformly within the interval [−b,b][−b,b], where
     * b = sqrt(6 /  (Hk + Hk+1)
     * where, Hk and Hk+1 are the sizes of the layers before and after the weight matrix.
     */
    function sample() {
        // return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
        //return Math.random() * 0.4 - 0.2;
        return (Math.floor(Math.random() * 201) - 100) / 100;   // -1 to 1
        // return Math.floor((Math.random() * ((b * 200) + 1)) - (b * 100)) / 100;  // don't hardcode to -1 to 1 as above
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
 * Wilson "class"
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
    
    // TOOD - Predefined or user defined. If user defined cannot save()?
    var activation;
    var activationPrime;
    
    var regress = false;
    
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
     * Labels for output data
     */
    var labels = [];
    
    /**
     * Sigmoid activation function
     * Returns a value between 0 and 1
     * 
     * @param number val - The value to apply the function to
     * @return number
     */
    function sigmoid(val){
       return 1 / (1 + Math.exp(-val));
    }
    
    /**
     * Sigmoid value to derivitive
     * 
     * @see: http://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
     * @param number val - The value to apply the function to
     * @return number
     */
    function sigmoidPrime(val) {
        return val * (1-val);
    }
    
    /**
     * Hyperbolic tan activation function
     * Returns a value between -1 and 1
     * 
     * @param number val - The value to apply the function to
     * @return number
     */
    function tanh(val) {
        return Math.tanh(val);
    }
    
    /**
     * Hyperbolic tan value to derivative
     * 
     * @see: http://math2.org/math/derivatives/more/hyperbolics.htm
     * @param number val - The value to apply the function to
     * @return number
     */
    function tanhPrime(val) {
        return val * (1 - val);
    }
    
    /**
     * Applies the Softmax function to all values in the given matrix
     * 
     * @see: https://en.wikipedia.org/wiki/Softmax_function
     * @param Matrix m - The matrix to apply Softmax to
     * @return Matrix
     */
    function softmax(m) {
        // get the data from the Matrix
        var arr = m.toArray();
        // setup a result array
        var res = [];
        
        // read each row from the matrix and apply Softmax to each value
        for (var i in arr) {
            var values = arr[i];
            var exponents = values.map(Math.exp),
            total = exponents.reduce(function (a, b) {
                return a + b;
            });
            
            // add the Softmax output values to the result array
            res.push(exponents.map(function (exp) { 
                return exp / total;
            }));
        }
        
        // return a new Matrix based on res
        return new Matrix(res);
    }
    
    /**
     * Helper function to log the state of the network at a given point
     */
    function log(id) {
        console.log('STATE AT ' + id);
        console.log('input > hidden weights', JSON.stringify(inputWeights.data(), null, 4));
        console.log('hidden values', JSON.stringify(hidden.data(), null, 4));
        console.log('hidden > output weights', JSON.stringify(hiddenWeights.data(), null, 4));
    }
    
    /**
     * Map predicted results to output classes/labels
     * 
     * @param Matrix prediction - The ouput of the predict() function
     * @return mixed
     */
    function getLabel(prediction) {
        // get array and read first (and only) entry
        prediction = prediction.toArray()[0];
        // find the index from this array with the max value. see: http://stackoverflow.com/questions/11301438/return-index-of-greatest-value-in-an-array
        var bestIdx = prediction.indexOf(Math.max.apply(Math, prediction));
        // return the corrosponding output/class label
        return {
            label: labels[bestIdx],
            score: prediction[bestIdx]
        };
    }
    
    /**
     * Appy configuration options to the network
     */
    function config() {
        hiddenNodes = opts.hiddenNodes || 3;    // number of hidden neurons
        iterations = opts.iterations || 10000;  // number of iterations
        learningRate = opts.learningRate || 0.1;
        // TODO: Not sure this works as expected...
        activation = opts.activation || (typeof opts.activation === 'string' ? opts.activation == 'sigmoid' ? sigmoid : opts.activation == 'tanh' ? tanh : sigmoid : sigmoid);    // activation function
        activationPrime = opts.activationPrime || (typeof opts.activation === 'string' ? opts.activation == 'sigmoidPrime' ? sigmoidPrime : opts.activation == 'tanhPrime' ? tanhPrime : sigmoidPrime : sigmoidPrime); // derivitive of activation function
    }
    
    /**
     * Forward propogation
     * Computes the network node values given the current set of weights.
     * Returns the resulting output(s)
     * 
     * @param Matrix inputs - The input data
     * @return Matrix
     */
    function forward(inputs) {
        // input > hidden
        // multiply the input weights by the inputs
        hidden = inputs.dot(inputWeights).map(function (val) {
            return activation(val);
        });
        
        // hidden > output
        // multiply the hidden weights by the hidden values and sum the resulting matrix (array)
        var output = hidden.dot(hiddenWeights);
        
        if (!regress) {
            output = softmax(output);
        } else {
            output = output.map(function (val) {
                return activation(val);
            });
        }
        
        // > output
        return output;
    }
    
    /**
     * Backward propogation
     * Uses Stochastic Gradient Descent to optimise
     * 
     * @param Matrix inputs - The training data
     * @param Matrix guess - The result of the forward propagation step
     * @param Matrix target - The target output(s)
     * @return Matrix
     */
    function backward(inputs, guess, target) {
        // output layer error
        var error = guess.minus(target);
        
        var outputDelta = error.mul(guess.map(activationPrime));
        
        // hidden layer error
        var hiddenLayerError = outputDelta.dot(hiddenWeights.trans());
        var hiddenLayerDelta = hiddenLayerError.mul(hidden.map(activationPrime));
        
        // adjust hidden > output weights
        hiddenWeights = hiddenWeights.minus(hidden.trans().dot(outputDelta).map(function (val) {
            return val * learningRate;
        }));
        
        // adjust input > hidden weights
        inputWeights = inputWeights.minus(inputs.trans().dot(hiddenLayerDelta).map(function (val) {
            return val * learningRate;
        }));
        
        // return the error
        return error;
    }
    
    // expose methods and properties
    return {
        /**
         * Given some inputs and target output(s), learn
         * 
         * @param array inputs - The input/training data
         * @param array target - The target output value(s)
         * @return void
         */
        learn: function (inputs, target, outputLayer, report) {
            // first configure the network
            config();
            
            // learn the outputs from the inputs
            
            /*// build the target/output layer based on the labels/classes given as targets
            var outputLayer = Matrix.zero(target.length, inputs.length).toArray();
            // get unique outputs
            function onlyUnique(value, index, self) { 
                return self.indexOf(value) === index;
            }
            
            // get the unique outputs
            var uniq = target.filter(onlyUnique);
            
            // map these to a binary representation
            // for example: ['red', 'green', 'blue'] => [[1,0,0], [0,1,0], [0,0,1]] or
            // ['red', 'green', 'red', blue'] => [[1,0,0], [0,1,0], [1,0,0], [0,0,1]]
            var outputMap = {};
            for (var idx in uniq) {
                if (!outputMap[uniq[idx]]) {
                    outputMap[uniq[idx]] = {
                        data: Matrix.zero(1, uniq.length).toArray()
                    }
                }
                
                outputMap[uniq[idx]].data[0][idx] = 1;
                labels[idx] = uniq[idx];
            }
            
            // set the output layer to have the correct binary representation for each entry
            for (var idx in target) {
                outputLayer[idx] = outputMap[target[idx]].data[0];
            }*/
            
            // set as matrix
            inputs = new Matrix(inputs);
            target = new Matrix(outputLayer);
            
            // initialize weights
            inputWeights = inputWeights.populate(inputs.toArray()[0].length, hiddenNodes);
            hiddenWeights = hiddenWeights.populate(hiddenNodes, target.toArray()[0].length);
            
            // learn yourself something
            for (var i=0; i < iterations; i++) {
                var guess = forward(inputs);
                var error = backward(inputs, guess, target);
                
                // output error margin every 1000 iterations
                if (i % 1000 == 0) {
                    var err = (function(err) {
                        // square the values
                        error = error.map(function (val) {
                            return val * val;
                        });
                        
                        // get the sum of the values
                        var sum = error.getSum();
                        
                        // return the mean (total / number of values)
                        return sum / error.toArray().length;
                    })(error);
                    
                    if (report) {
                        console.log('Error after ', i, 'iterations', err);
                    }
                    
                    if (err < 0.00005) {
                        console.log('Minimum error reached after', i, 'iterations');
                        break;
                    }
                }
            }
        },
        classify: function (inputs, target, report) {
            // build the target/output layer based on the labels/classes given as targets
            var outputLayer = Matrix.zero(target.length, inputs.length).toArray();
            // get unique outputs
            function onlyUnique(value, index, self) { 
                return self.indexOf(value) === index;
            }
            
            // get the unique outputs
            var uniq = target.filter(onlyUnique);
            
            // map these to a binary representation
            // for example: ['red', 'green', 'blue'] => [[1,0,0], [0,1,0], [0,0,1]] or
            // ['red', 'green', 'red', blue'] => [[1,0,0], [0,1,0], [1,0,0], [0,0,1]]
            var outputMap = {};
            for (var idx in uniq) {
                if (!outputMap[uniq[idx]]) {
                    outputMap[uniq[idx]] = {
                        data: Matrix.zero(1, uniq.length).toArray()
                    }
                }
                
                outputMap[uniq[idx]].data[0][idx] = 1;
                labels[idx] = uniq[idx];
            }
            
            // set the output layer to have the correct binary representation for each entry
            for (var idx in target) {
                outputLayer[idx] = outputMap[target[idx]].data[0];
            }
            
            regress = false;
            this.learn(inputs, target, outputLayer, report);
        },
        regress: function (inputs, target, report) {
            // build the target/output layer based on the labels/classes given as targets
            var outputLayer = [];
            for (var i in target) {
                outputLayer[i] = [target[i]];
            };
            
            regress = true;
            this.learn(inputs, target, outputLayer, report);
        },
        /**
         * Given an input value(s) predict the ouput(s)
         * 
         * @param array input - The input
         * @param float expected - The expected output
         * @return float
         */
        predict: function (input, expected) {
            // TODO: Support multiple inputs and outputs
            // first configure the network
            config();
            
            // predict the output from the input
            var prediction = forward(new Matrix(input));
            
            // console.log('predicted', (prediction).toArray(), getLabel(prediction), 'expected', expected);
            return {
                scores: prediction.toArray(),
                best: {
                    label: getLabel(prediction).label,
                    score: getLabel(prediction).score
                }
            };
        },
        /**
         * Configure property(s) of the network after initialisation
         * 
         * @param object conf - JSON object holding config data
         * @return void
         */
        configure: function (conf) {
            // update any current config with values passed
            for (var op in conf) {
                opts[op] = conf[op];
            }
        },
        /**
         * Save the network config and state
         * 
         * @return string
         */
        save: function () {
            // return a JSON string of the weights/parameters
            return JSON.stringify({
                hyperParams: {
                    hiddenNodes: opts.hiddenNodes,
                    iterations: opts.iterations,
                    learningRate: opts.learningRate
                },
                weights: [
                    inputWeights.toArray(),
                    hiddenWeights.toArray()
                ]
            });
        },
        /**
         * Configure the network with a previously saved instance
         * 
         * @param string config - A JSON string representing the network state
         * @return void
         */
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