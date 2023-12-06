`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps

module tb_arithmetic;

    reg [7:0] x1,x2,x3;
    wire [7:0] y;

    // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns
    localparam period = 2;  

    arithmetic UUT (.y(y), .x1(x1), .x2(x2), .x3(x3));
    
initial begin
        // Test case 1
        x1 = 8'b00000000;
        x2 = 8'b00000000;
        x3 = 8'b00000000;
        #period;
        if (y !== 8'b00000000) begin
            $display("Test 1 failed");
            $display("x1=%b, x2=%b, x3=%b, y=%b", x1, x2, x3, y);
            $finish;
        end else
            $display("x1=%b, x2=%b, x3=%b, y=%b", x1, x2, x3, y);

    	// Test case 2
        x1 = 8'b00000100;
        x2 = 8'b00000101;
        x3 = 8'b00000010;
        #period;
        if (y !== 8'b00011110) begin
            $display("Test 2 failed");
            $display("x1=%b, x2=%b, x3=%b, y=%b", x1, x2, x3, y);
            $finish;
        end else
            $display("x1=%b, x2=%b, x3=%b, y=%b", x1, x2, x3, y);
    

        $display("all tests passed");
        $finish;

        end

endmodule