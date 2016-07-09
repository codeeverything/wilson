
// get wilson
var wilson = require('./wilson.js')();

/**
 * Helper functions for running tests and outputting results
 */
var regression = false;
 
function classify(name, inputs, targets, report) {
    regression = false;
    console.log('');
    console.log('************************************************************');
    console.log('Running "' + name + '" test...');
    console.log('    Learning...');
    console.log('    Inputs: ' + JSON.stringify(inputs.slice(0, 5)) + '...');
    console.log('    Targets: ' + JSON.stringify(targets.slice(0, 5)) + '...');
    wilson.classify(inputs, targets, report);
}

function regress(name, inputs, targets, report) {
    regression = true;
    console.log('');
    console.log('************************************************************');
    console.log('Running "' + name + '" test...');
    console.log('    Learning...');
    console.log('    Inputs: ' + JSON.stringify(inputs.slice(0, 5)) + '...');
    console.log('    Targets: ' + JSON.stringify(targets.slice(0, 5)) + '...');
    wilson.regress(inputs, targets, report);
}

function predict(input, expected) {
    var p = wilson.predict(input);
    console.log('Predicting...');
    console.log('    Input: ' + JSON.stringify(input.slice(0, 5)) + '...');
    console.log('    Expected: ' + expected);
    if (!regression) {
        console.log('    Output: ' + p.best.label + ' (' + ((p.best.score * 100).toFixed(2)) + '% confidence)');
    } else {
        console.log('    Output: ' + p.best.score);
    }
}

/**
 * Tests
 */

// learn hours study + hours sleep relation to test scores: regression
// https://www.youtube.com/watch?v=bxe2T-V8XRs
wilson.configure({
    learningRate: 0.5
});

regress('STUDY: Regression', [
    [3,5],
    [5,1],
    [10,2],
], [
    0.75,
    0.82,
    0.93,
], true);

predict([[3,5]], 0.75);
predict([[5,1]], 0.82);
predict([[10,2]], 0.93);

// learn XOR truth table: regression
wilson.configure({
    learningRate: 0.5
});

regress('XOR: Regression', [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], [
    0,
    1,
    1,
    0
], true);

predict([[0,0]], 0);
predict([[0,1]], 1);
predict([[1,0]], 1);
predict([[1,1]], 0);

// learn XOR truth table: classification
wilson.configure({
    learningRate: 0.1
});

classify('XOR: Classification', [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], [
    0,
    1,
    1,
    0
], true);

predict([[0,0]], 0);
predict([[0,1]], 1);
predict([[1,0]], 1);
predict([[1,1]], 0);


// learn RGB
classify('RGB: Classification', [
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
], [
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
    'red',
    'green',
    'blue',
]);

// test - known values
predict([[1,1,1]], 'red');
predict([[0,1,1]], 'green');
predict([[0,0,1]], 'blue');

// test - unknown values
predict([[1.1,1.1,1.1]], 'red');
predict([[0,0.5,0.5]], 'green');


// train IRIS
wilson.configure({
    learningRate: 0.05,
    hiddenNodes: 3,
    iterations: 20000
});

classify('IRIS: Classification', [
    [5.1,3.5,1.4,0.2],
    [4.9,3,1.4,0.2],
    [4.7,3.2,1.3,0.2],
    [4.6,3.1,1.5,0.2],
    [5,3.6,1.4,0.2],
    [5.4,3.9,1.7,0.4],
    [4.6,3.4,1.4,0.3],
    [5,3.4,1.5,0.2],
    [4.4,2.9,1.4,0.2],
    [4.9,3.1,1.5,0.1],
    [5.4,3.7,1.5,0.2],
    [4.8,3.4,1.6,0.2],
    [4.8,3,1.4,0.1],
    [4.3,3,1.1,0.1],
    [5.8,4,1.2,0.2],
    [5.7,4.4,1.5,0.4],
    [5.4,3.9,1.3,0.4],
    [5.1,3.5,1.4,0.3],
    [5.7,3.8,1.7,0.3],
    [5.1,3.8,1.5,0.3],
    [5.4,3.4,1.7,0.2],
    [5.1,3.7,1.5,0.4],
    [4.6,3.6,1,0.2],
    [5.1,3.3,1.7,0.5],
    [4.8,3.4,1.9,0.2],
    [5,3,1.6,0.2],
    [5,3.4,1.6,0.4],
    [5.2,3.5,1.5,0.2],
    [5.2,3.4,1.4,0.2],
    [4.7,3.2,1.6,0.2],
    [4.8,3.1,1.6,0.2],
    [5.4,3.4,1.5,0.4],
    [5.2,4.1,1.5,0.1],
    [5.5,4.2,1.4,0.2],
    [4.9,3.1,1.5,0.1],
    [5,3.2,1.2,0.2],
    [5.5,3.5,1.3,0.2],
    [4.9,3.1,1.5,0.1],
    [4.4,3,1.3,0.2],
    [5.1,3.4,1.5,0.2],
    [5,3.5,1.3,0.3],
    [4.5,2.3,1.3,0.3],
    [4.4,3.2,1.3,0.2],
    [5,3.5,1.6,0.6],
    [5.1,3.8,1.9,0.4],
    [4.8,3,1.4,0.3],
    [5.1,3.8,1.6,0.2],
    [4.6,3.2,1.4,0.2],
    [5.3,3.7,1.5,0.2],
    [5,3.3,1.4,0.2],
    [7,3.2,4.7,1.4],
    [6.4,3.2,4.5,1.5],
    [6.9,3.1,4.9,1.5],
    [5.5,2.3,4,1.3],
    [6.5,2.8,4.6,1.5],
    [5.7,2.8,4.5,1.3],
    [6.3,3.3,4.7,1.6],
    [4.9,2.4,3.3,1],
    [6.6,2.9,4.6,1.3],
    [5.2,2.7,3.9,1.4],
    [5,2,3.5,1],
    [5.9,3,4.2,1.5],
    [6,2.2,4,1],
    [6.1,2.9,4.7,1.4],
    [5.6,2.9,3.6,1.3],
    [6.7,3.1,4.4,1.4],
    [5.6,3,4.5,1.5],
    [5.8,2.7,4.1,1],
    [6.2,2.2,4.5,1.5],
    [5.6,2.5,3.9,1.1],
    [5.9,3.2,4.8,1.8],
    [6.1,2.8,4,1.3],
    [6.3,2.5,4.9,1.5],
    [6.1,2.8,4.7,1.2],
    [6.4,2.9,4.3,1.3],
    [6.6,3,4.4,1.4],
    [6.8,2.8,4.8,1.4],
    [6.7,3,5,1.7],
    [6,2.9,4.5,1.5],
    [5.7,2.6,3.5,1],
    [5.5,2.4,3.8,1.1],
    [5.5,2.4,3.7,1],
    [5.8,2.7,3.9,1.2],
    [6,2.7,5.1,1.6],
    [5.4,3,4.5,1.5],
    [6,3.4,4.5,1.6],
    [6.7,3.1,4.7,1.5],
    [6.3,2.3,4.4,1.3],
    [5.6,3,4.1,1.3],
    [5.5,2.5,4,1.3],
    [5.5,2.6,4.4,1.2],
    [6.1,3,4.6,1.4],
    [5.8,2.6,4,1.2],
    [5,2.3,3.3,1],
    [5.6,2.7,4.2,1.3],
    [5.7,3,4.2,1.2],
    [5.7,2.9,4.2,1.3],
    [6.2,2.9,4.3,1.3],
    [5.1,2.5,3,1.1],
    [5.7,2.8,4.1,1.3],
    [6.3,3.3,6,2.5],
    [5.8,2.7,5.1,1.9],
    [7.1,3,5.9,2.1],
    [6.3,2.9,5.6,1.8],
    [6.5,3,5.8,2.2],
    [7.6,3,6.6,2.1],
    [4.9,2.5,4.5,1.7],
    [7.3,2.9,6.3,1.8],
    [6.7,2.5,5.8,1.8],
    [7.2,3.6,6.1,2.5],
    [6.5,3.2,5.1,2],
    [6.4,2.7,5.3,1.9],
    [6.8,3,5.5,2.1],
    [5.7,2.5,5,2],
    [5.8,2.8,5.1,2.4],
    [6.4,3.2,5.3,2.3],
    [6.5,3,5.5,1.8],
    [7.7,3.8,6.7,2.2],
    [7.7,2.6,6.9,2.3],
    [6,2.2,5,1.5],
    [6.9,3.2,5.7,2.3],
    [5.6,2.8,4.9,2],
    [7.7,2.8,6.7,2],
    [6.3,2.7,4.9,1.8],
    [6.7,3.3,5.7,2.1],
    [7.2,3.2,6,1.8],
    [6.2,2.8,4.8,1.8],
    [6.1,3,4.9,1.8],
    [6.4,2.8,5.6,2.1],
    [7.2,3,5.8,1.6],
    [7.4,2.8,6.1,1.9],
    [7.9,3.8,6.4,2],
    [6.4,2.8,5.6,2.2],
    [6.3,2.8,5.1,1.5],
    [6.1,2.6,5.6,1.4],
    [7.7,3,6.1,2.3],
    [6.3,3.4,5.6,2.4],
    [6.4,3.1,5.5,1.8],
    [6,3,4.8,1.8],
    [6.9,3.1,5.4,2.1],
    [6.7,3.1,5.6,2.4],
    [6.9,3.1,5.1,2.3],
    [5.8,2.7,5.1,1.9],
    [6.8,3.2,5.9,2.3],
    [6.7,3.3,5.7,2.5],
    [6.7,3,5.2,2.3],
    [6.3,2.5,5,1.9],
    [6.5,3,5.2,2],
    // [6.2,3.4,5.4,2.3],
    // [5.9,3,5.1,1.8  ]    // we'll test with this one
], [
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
    // "Iris-virginica",
    // "Iris-virginica"    // we'll test with this one
], true);

// trained 
predict([
    [5.1,3.5,1.4,0.2]
], 'Iris-setosa');

// unknown
predict([
    [6.2,3.4,5.4,2.3]
], 'Iris-virginica');

// unknown
predict([
    [6.2,3.4,5.4,2.3]
], 'Iris-virginica');