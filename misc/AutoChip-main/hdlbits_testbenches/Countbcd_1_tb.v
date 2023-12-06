`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps

module top_module_tb;

    // duration for each bit = 20 * timescale = 20 * 1 ns  = 20ns
    localparam period = 20;

    reg clk;
    reg reset;

    wire [2:0] ena;
    wire [15:0] q;


    integer mismatch_count;

    top_module UUT (.clk(clk), .reset(reset), .ena(ena), .q(q));

    initial // clk generation
    begin
        clk = 0;
        forever begin
            #(period/2);
            clk = ~clk;
        end
    end

    initial begin
        mismatch_count = 0;

        // Tick 0: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000010
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000010)) begin
            $display("Mismatch at index 0: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000000010);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 0 passed!");
        end

        // Tick 1: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000011
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000011)) begin
            $display("Mismatch at index 1: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000000011);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 1 passed!");
        end

        // Tick 2: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000011
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000011)) begin
            $display("Mismatch at index 2: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000000011);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 2 passed!");
        end

        // Tick 3: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000100
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000100)) begin
            $display("Mismatch at index 3: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000000100);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 3 passed!");
        end

        // Tick 4: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000100
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000100)) begin
            $display("Mismatch at index 4: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000000100);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 4 passed!");
        end

        // Tick 5: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000101
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000101)) begin
            $display("Mismatch at index 5: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000000101);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 5 passed!");
        end

        // Tick 6: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000101
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000101)) begin
            $display("Mismatch at index 6: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000000101);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 6 passed!");
        end

        // Tick 7: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000110
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000110)) begin
            $display("Mismatch at index 7: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000000110);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 7 passed!");
        end

        // Tick 8: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000110
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000110)) begin
            $display("Mismatch at index 8: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000000110);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 8 passed!");
        end

        // Tick 9: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000111
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000111)) begin
            $display("Mismatch at index 9: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000000111);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 9 passed!");
        end

        // Tick 10: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000000111
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000000111)) begin
            $display("Mismatch at index 10: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000000111);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 10 passed!");
        end

        // Tick 11: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000001000
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000001000)) begin
            $display("Mismatch at index 11: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000001000);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 11 passed!");
        end

        // Tick 12: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000001000
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000001000)) begin
            $display("Mismatch at index 12: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000001000);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 12 passed!");
        end

        // Tick 13: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b001, 16'b0000000000001001
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b001 && q === 16'b0000000000001001)) begin
            $display("Mismatch at index 13: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b001, 16'b0000000000001001);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 13 passed!");
        end

        // Tick 14: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b001, 16'b0000000000001001
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b001 && q === 16'b0000000000001001)) begin
            $display("Mismatch at index 14: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b001, 16'b0000000000001001);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 14 passed!");
        end

        // Tick 15: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010000
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010000)) begin
            $display("Mismatch at index 15: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000010000);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 15 passed!");
        end

        // Tick 16: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010000
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010000)) begin
            $display("Mismatch at index 16: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000010000);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 16 passed!");
        end

        // Tick 17: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010001
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010001)) begin
            $display("Mismatch at index 17: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000010001);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 17 passed!");
        end

        // Tick 18: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010001
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010001)) begin
            $display("Mismatch at index 18: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000010001);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 18 passed!");
        end

        // Tick 19: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010010
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010010)) begin
            $display("Mismatch at index 19: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000010010);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 19 passed!");
        end

        // Tick 20: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010010
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010010)) begin
            $display("Mismatch at index 20: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000010010);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 20 passed!");
        end

        // Tick 21: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010011
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010011)) begin
            $display("Mismatch at index 21: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000010011);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 21 passed!");
        end

        // Tick 22: Inputs = 1'b0, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010011
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010011)) begin
            $display("Mismatch at index 22: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b0, 1'b0, ena, q, 3'b000, 16'b0000000000010011);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 22 passed!");
        end

        // Tick 23: Inputs = 1'b1, 1'b0, Generated = ena, q, Reference = 3'b000, 16'b0000000000010100
        reset = 1'b0; // Set input values
        #period;
        if (!(ena === 3'b000 && q === 16'b0000000000010100)) begin
            $display("Mismatch at index 23: Inputs = [%b, %b], Generated = [%b, %b], Reference = [%b, %b]", 1'b1, 1'b0, ena, q, 3'b000, 16'b0000000000010100);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 23 passed!");
        end

        if (mismatch_count == 0)
            $display("All tests passed!");
        else
            $display("%0d mismatches out of %0d total tests.", mismatch_count, 24);
        $finish;
    end

endmodule