// Design a signed adder that adds two 8-bit 2's complement numbers, a[7:0] and b[7:0]. 
// These numbers are added to produce s[7:0]. Also compute whether a (signed) overflow has occurred.

module signed_adder(input [7:0] a, input [7:0] b, output [7:0] s, output overflow ); 