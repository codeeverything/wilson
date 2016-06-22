// Neo - A small matrix manipulation class. Strongly tied to the needs of Wilson
// TODO: All methods, except data(), should return a Matrix object
var Matrix = function (dataArray) {
    var data = dataArray;
    
    return {
        data: function () {
            return data;
        },
        add: function (m2) {
            m2 = m2.data();
            var res = [];
            for (var i=0; i < data.length; i++) {
                res[i] = [];
                for (var j=0; j < data[i].length; j++) {
                    res[i][j] = data[i][j] + m2[i][j];
                }
            }
            
            return res;
        },
        subtract: function (m2) {
            m2 = m2.data();
            var res = [];
            for (var i=0; i < data.length; i++) {
                res[i] = [];
                for (var j=0; j < data[i].length; j++) {
                    res[i][j] = data[i][j] - m2[i][j];
                }
            }
            
            return res;
        },
        multiply: function (m2) {
            var result = [];

            // read each row from m2
            for (var i = 0; i < m2.data().length; i++) {
                result[i] = [];
                
                // read each column from this
                for (var j = 0; j < data[0].length; j++) {
                    var sum = 0;
                    
                    // read each row from this
                    for (var k = 0; k < data.length; k++) {
                        sum += data[k][j] * m2.data()[i][k];
                    }
                    
                    result[i][j] = sum;
                }
            }
            
            return new Matrix(result);
        },
        transform: function (callback) {
            return new Matrix(data.map(function (row) {
                return row.map(callback);
            }));
        },
        transpose: function () {
            // see: http://stackoverflow.com/questions/17428587/transposing-a-2d-array-in-javascript
            return new Matrix(data[0].map(function(col, i) { 
                return data.map(function(row) { 
                    return row[i];
                });
            }));
        },
        populate: function (x, y) {
            var res = [];
            for (var i=0; i < x; i++) {
                res[i] = [];
                for (var j=0; j < y; j++) {
                    res[i][j] = Math.random();
                }
            }
            
            data = res;
        }
    }
};

module.exports = Matrix;