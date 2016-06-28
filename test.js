
// train OR
var wilson = require('./wilson.js')();

// learn OR
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

// test
wilson.predict([[1,1,1]], 'red');
wilson.predict([[0,1,1]], 'green');
wilson.predict([[0,0,1]], 'blue');
// new
wilson.predict([[1.1,1.1,1.1]], 'red');
wilson.predict([[0,0.5,0.5]], 'green');

// learn OR
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