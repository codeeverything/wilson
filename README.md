# Wilson

IBM Watson's very distant cousin.

Wilson is an experiment in artificial neural networks. It's a work in progress, but has been tested to work on simple problems like the OR truth table.

I have also tested "real" datasets such as Iris, but Wilson is much less stable with this for some reason (suggestions welcomed!) :)

Wilson is implemented in NodeJS, but there's no reason you couldn't fairly easily make it work in the browser, or port to another language (for example, Wilson is largely based off of other work written in Python).

### Tests/examples

Run ```$ node test.js```

### Learning the OR truth table

We can learn the OR truth table as follows:

```
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
wilson.predict([[1,1]], 1); // ~0.97
wilson.predict([[0,1]], 1); // ~0.97
wilson.predict([[1,0]], 1); // ~0.97
wilson.predict([[0,0]], 0); // ~0
```

## License

MIT