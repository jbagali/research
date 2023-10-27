// This is a 2-to-1 multiplexer.
// When sel=0, assign a to out. 
// When sel=1, assign b to out.
module mux( 
input [4:0] a, b,
input sel,
output [4:0] out );