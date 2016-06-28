
// get wilson
var wilson = require('./wilson.js')();

// learn OR truth table
wilson.learn([
    [1,1],
    [0,0],
    [0,1],
    [1,0]
], [
    1,
    0,
    1,
    1
]);

// test
wilson.predict([[1,1]], 1);
wilson.predict([[0,1]], 1);
wilson.predict([[1,0]], 1);
wilson.predict([[0,0]], 0);

// learn RGB
wilson.learn([
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
wilson.predict([[1,1,1]], 'red');
wilson.predict([[0,1,1]], 'green');
wilson.predict([[0,0,1]], 'blue');

// test - unknown values
wilson.predict([[1.1,1.1,1.1]], 'red');
wilson.predict([[0,0.5,0.5]], 'green');
