
// get wilson
var wilson = require('./wilson.js')();

/**
 * Helper functions for running tests and outputting results
 */
function test(name, inputs, targets, report) {
    console.log('');
    console.log('************************************************************');
    console.log('Running "' + name + '" test...');
    console.log('    Learning...');
    console.log('    Inputs: ' + JSON.stringify(inputs.slice(0, 5)) + '...');
    console.log('    Targets: ' + JSON.stringify(targets.slice(0, 5)) + '...');
    wilson.classify(inputs, targets, report);
}

function predict(input, expected) {
    var p = wilson.predict(input);
    console.log('Predicting...');
    console.log('    Input: ' + JSON.stringify(input.slice(0, 5)) + '...');
    console.log('    Expected: ' + expected);
    console.log('    Output: ' + p.best.label + ' (' + ((p.best.score * 100).toFixed(2)) + '% confidence)');
}

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

test('OCR: Classification', [
    a,
    b,
    c
], [
    'a',
    'b',
    'c'
], true);

/**
 * Predict the letter C, even with a pixel off.
 */

predict(character(
  '.######' +
  '#......' +
  '#......' +
  '#.#....' +
  '#......' +
  '##.....' +
  '######.'
), 'c');

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