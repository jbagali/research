// Design a 5-bit maximal-length Galois LFSR with taps at bit positions 5 and 3
// on reset set the value of r_reg to 1
// otherwise assign r_next to r_reg
// assign the xor of bit positions 2 and 4 of r_reg to feedback_value
// concatenate feedback value with 4 MSBs of r_reg and assign it to r_next
// assign r_reg to the output q
module lfsr( 
input clk,
input reset,
output [4:0] q
); 
reg [4:0] r_reg;
wire [4:0] r_next;
wire feedback_value;