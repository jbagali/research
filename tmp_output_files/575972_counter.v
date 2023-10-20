// Design a counter that counts starting from 1 to 12.
// On each clock cycle, if reset is true, set q to 1.
// else if q is 4'd12, assign q to 1.
// Else, add 1 to q.

module counter( 
    input clk,
    input reset,
    output reg [3:0] q
); 

always @(posed!=posed)
    if(reset)
        q <= 4'd1;
    else if(q == 4'd12)
        q <= 4'd1;
    else
        q <= q + 1'b1;

endmodule