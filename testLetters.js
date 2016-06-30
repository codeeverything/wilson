
// get wilson
var wilson = require('./wilson.js')();

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

wilson.learn([
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

var result = wilson.predict(character(
  '.######' +
  '#......' +
  '#......' +
  '#.#....' +
  '#......' +
  '##.....' +
  '######.'
), 'c');

console.log(result); // ~ 0.5

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