//Design a Verilog module for a MAC (Multiplier and Accumulator) unit.
//The MAC unit takes two inputs, A and B, and multiplies them.
//The result of the multiplication is accumulated with a third input, C.
//The accumulated result is stored in an output register, P.
//The module should have a control input, clk (clock), to synchronize the operations.
//The module should have a control input, rst (reset), to reset the accumulated result.

module mac_unit_16(
    input wire [15:0] multiplier,
    input wire [15:0] multiplicand,
    input wire clk,
    input wire rst,
    output reg [31:0] result
);
