// This is a decade counter that counts from 0 through 9, inclusive. It counts only when slowena is high.
// On the positive edge of the clock:
// if reset is high, reset the output q to 0. 
// Otherwise, only increment the output q if the slowena input is high and q is not 9.
module count_slow(input clk, input slowena, input reset, output reg [3:0] q);