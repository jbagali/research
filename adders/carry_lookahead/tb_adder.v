`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps

module tb_adder;

    reg [3:0] in1, in2;
    reg cin;
    wire [3:0] sum;
    wire cout;

    // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns
    localparam period = 2;  

    adder UUT (.sum(sum), .cout(cout), .in1(in1), .in2(in2), .cin(cin));
    
initial begin
        // Test case 1: 0 + 0 + 0 = 0
        in1 = 4'b0000;
        in2 = 4'b0000;
        cin = 0;
        #period;
        if (cout !== 0 || sum !== 4'b0000) begin
            $display("Test 1 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 2: 0 + 0 + 1 = 0
        in1 = 4'b0000;
        in2 = 4'b0000;
        cin = 1;
        #period;
        if (cout !== 0 || sum !== 4'b0001) begin
            $display("Test 2 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 3: 0 + 1 + 0 = 1
        in1 = 4'b0000;
        in2 = 4'b0001;
        cin = 0;
        #period;
        if (cout !== 0 || sum !== 4'b0001) begin
            $display("Test 3 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        
        // Test case 4: 0 + 1 + 1 = 2
        in1 = 4'b0000;
        in2 = 4'b0001;
        cin = 1;
        #period;
        if (cout !== 0 || sum !== 4'b0010) begin
            $display("Test 4 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);


        // Test case 5: 0 + 15 + 0 = 15
        in1 = 4'b0000;
        in2 = 4'b1111;
        cin = 0;
        #period;
        if (cout !== 0 || sum !== 4'b1111) begin
            $display("Test 5 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 6: 0 + 15 + 1 = 15 (Carry)
        in1 = 4'b0000;
        in2 = 4'b1111;
        cin = 1;
        #period;
        if (cout !== 1 || sum !== 4'b0000) begin
            $display("Test 6 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

            // Test case 7: 15 + 0 + 0 = 15
        in1 = 4'b1111;
        in2 = 4'b0000;
        cin = 0;
        #period;
        if (cout !== 0 || sum !== 4'b1111) begin
            $display("Test 7 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 8: 15 + 0 + 1 = 15 (Carry)
        in1 = 4'b1111;
        in2 = 4'b0000;
        cin = 1;
        #period;
        if (cout !== 1 || sum !== 4'b0000) begin
            $display("Test 8 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 9: 15 + 1 + 1 = 1 (With Carry-Out)
        in1 = 4'b1111;
        in2 = 4'b0001;
        cin = 1;
        #period;
        if (cout !== 1 || sum !== 4'b0001) begin
            $display("Test 9 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 10: 15 + 15 + 0 = 1110 (Carry)
        in1 = 4'b1111;
        in2 = 4'b1111;
        cin = 0;
        #period;
        if (cout !== 1 || sum !== 4'b1110) begin
            $display("Test 10 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 11: 15 + 15 + 1 = 1111 (Carry)
        in1 = 4'b1111;
        in2 = 4'b1111;
        cin = 1;
        #period;
        if (cout !== 1 || sum !== 4'b1111) begin
            $display("Test 11 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 12: 7 + 8 + 0 = 15 (No Carry-Out)
        in1 = 4'b0111;
        in2 = 4'b1000;
        cin = 0;
        #period;
        if (cout !== 0 || sum !== 4'b1111) begin
            $display("Test 12 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 13: 7 + 8 + 1 = 0 (Carry)
        in1 = 4'b0111;
        in2 = 4'b1000;
        cin = 1;
        #period;
        if (cout !== 1 || sum !== 4'b0000) begin
            $display("Test 13 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);


        // Test case 14: 2 + 5 + 0 = 7
        in1 = 4'b0010;
        in2 = 4'b0101;
        cin = 0;
        #period;
        if (cout !== 0 || sum !== 4'b0111) begin
            $display("Test 14 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);

        // Test case 15: 2 + 5 + 1 = 8
        in1 = 4'b0010;
        in2 = 4'b0101;
        cin = 1;
        #period;
        if (cout !== 0 || sum !== 4'b1000) begin
            $display("Test 15 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);
        // Add more test cases here if needed.

	// Test case 16: 6 + 13 + 0 = 19
        in1 = 4'b0110;
        in2 = 4'b1101;
        cin = 0;
        #period;
        if (cout !== 1 || sum !== 4'b0011) begin
            $display("Test 16 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);
        // Add more test cases here if needed.

	// Test case 17: 6 + 13 + 1 = 20
        in1 = 4'b0110;
        in2 = 4'b1101;
        cin = 1;
        #period;
        if (cout !== 1 || sum !== 4'b0100) begin
            $display("Test 17 failed");
            $finish;
        end else
            $display("in1=%b, in2=%b, cin=%b, cout=%b, sum=%b", in1, in2, cin, cout, sum);
        // Add more test cases here if needed.


        $display("all tests passed");
        $finish;

        end


endmodule