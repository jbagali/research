// This is a counter that counts from 1 to 12
    module counter(
    input clk,
    input reset,
    output reg [3:0] q
    );
    
    always @(posedge clk)
        if(reset)
            q <= 4'b0000;
        else
            q <= q + 1;
endmodule