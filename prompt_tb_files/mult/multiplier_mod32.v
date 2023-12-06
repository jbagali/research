//Please finish designing the 32 bit multiplier module (multiplier_32) below.
//There are two 32 bit inputs (A and B) and a 64 bit output (product).
//The module should utilize the inputs (A and B) to determine the output product correctly.
//The module should also utilize the completed 16 bit multiplier (listed below) within the 32 bit module in implementation.

module multiplier_16(output [31:0] product, input [15:0] A, B);
	assign product = A * B;
endmodule

module multiplier_32(output [63:0] product, input [31:0] A, B);