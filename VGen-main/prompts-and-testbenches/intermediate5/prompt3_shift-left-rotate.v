// This is a shift left and rotate operation
// when load is high, load data to out
// when load is low, shift left and rotate the register out by amount bits
module left_rotate(input clk,input reset,input [2:0] amount,input [7:0] data,input load,output reg [7:0] out);