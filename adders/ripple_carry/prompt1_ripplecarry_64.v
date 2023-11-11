//Design a 64-bit adder.
//There are two 64 bit inputs (in1 and in2) and a single carry-in input bit, cin.
//The outputs are a 64 bit value, [63:0] sum, and a single carry-out output bit, cout.
//The module should add the inputs (in1 and in2) and the cin bit to obtain the output sum and cout values.
module adder_64(output [63:0] sum,
                      output cout,
                      input [63:0] in1, in2,
                      input cin);