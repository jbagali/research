`timescale 1 ns/10 ps

module tb_adder_8;

  reg [7:0] in1, in2;
  reg cin;
  wire [7:0] sum;
  wire cout;

  // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns
  localparam period = 2;  

  // Instantiate the 8-bit carry-select adder
  adder_8 uut(
    .sum(sum),
    .cout(cout),
    .in1(in1),
    .in2(in2),
    .cin(cin)
  );

  // Clock generation
  reg clk;
  always #5 clk = ~clk;

  // Test cases
  initial begin
    // Test case 1: 0 + 0 + 0 = 0
    in1 = 8'h00;
    in2 = 8'h0000;
    cin = 0;
    #period;
    if (cout !== 0 || sum !== 8'h00) begin
      $display("Test 1 failed");
      $finish;
    end else
      $display("Test 1 passed");

    // Test case 2: 1 + 1 + 1 = 3
    in1 = 8'h01;
    in2 = 8'h01;
    cin = 1;
    #period;
    if (cout !== 0 || sum !== 8'h03) begin
      $display("Test 2 failed");
      $finish;
    end else
      $display("Test 2 passed");

    // Test case 3: Max positive value + 1 + 0 = Overflow (check cout)
    in1 = 8'hFF;
    in2 = 8'h01;
    cin = 0;
    #period;
    if (cout !== 1 || sum !== 8'h00) begin
      $display("Test 3 failed");
      $finish;
    end else
      $display("Test 3 passed");

    // Test case 4: Max positive value + 1 + 1 = Overflow (check cout)
    in1 = 8'hFF;
    in2 = 8'h01;
    cin = 1;
    #period;
    if (cout !== 1 || sum !== 8'h01) begin
      $display("Test 4 failed");
      $finish;
    end else
      $display("Test 4 passed");

    // Test case 5: Max positive value + Max positive value + 0 = Overflow (check cout)
    in1 = 8'hFF;
    in2 = 8'hFF;
    cin = 0;
    #period;
    if (cout !== 1 || sum !== 8'hFE) begin
      $display("Test 5 failed");
      $finish;
    end else
      $display("Test 5 passed");

    // Test case 6: Max positive value + Max positive value + 1 = Overflow (check cout)
    in1 = 8'hFF;
    in2 = 8'hFF;
    cin = 1;
    #period;
    if (cout !== 1 || sum !== 8'hFF) begin
      $display("Test 6 failed");
      $finish;
    end else
      $display("Test 6 passed");

    // Test case 7: Alternating 1's and 0's - No Overflow
    in1 = 8'hAA;
    in2 = 8'h55;
    cin = 0;
    #period;
    if (cout !== 0 || sum !== 8'hFF) begin
      $display("Test 7 failed");
      $finish;
    end else
      $display("Test 7 passed");

    // Test case 8: Alternating 1's and 0's with Carry
    in1 = 8'hAA;
    in2 = 8'h55;
    cin = 1;
    #period;
    if (cout !== 1 || sum !== 8'h00) begin
      $display("Test 8 failed");
      $finish;
    end else
      $display("Test 8 passed");

    // Add more test cases as needed.

    $display("all tests passed");
    $finish;
  end

  // Monitor outputs
  always @(posedge clk) begin
    $display("Time: %t, Cin: %b, In1: %h, In2: %h, Sum: %h, Cout: %b", $time, cin, in1, in2, sum, cout);
  end

endmodule