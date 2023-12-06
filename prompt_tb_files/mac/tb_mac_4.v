`timescale 1 ns/1 ps

module tb_mac_4;

  reg clk, reset;
  reg [3:0] A, B; // Assuming N=4 for simplicity
  reg [7:0] expected_accumulator;
  wire [7:0] accumulator;

  // Instantiate the MACUnit module
  mac_4 uut (
    .clk(clk),
    .reset(reset),
    .A(A),
    .B(B),
    .accumulator(accumulator)
  );

  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // Test cases
  initial begin
    // Test case 1: Reset and verify initial state
    reset = 1;
    A = 4'b0000;
    B = 4'b0000;
    #15 reset = 0;

    // Wait for a few clock cycles
    #15;
    @(posedge clk);
    // Check if the accumulator is in the initial state
    if (accumulator !== 8'b00000000) begin
      $display("Test 1 failed");
      $finish;
    end else
      $display("Test 1 passed");

    // Test case 2: Perform a multiplication and check the accumulator
    A = 4'b0010; // 2
    B = 4'b0011; // 3
    

    expected_accumulator = 8'b00000110; // 6
    @(posedge clk);
    if (accumulator !== expected_accumulator) begin
      $display("Test 2 failed");
      $finish;
    end else
      $display("Test 2 passed");

    // Test case 3: Perform another multiplication with reset
    reset = 1;
    @(posedge clk);
    A = 4'b1101; // 13
    B = 4'b0100; // 4
    reset = 0;
    expected_accumulator = (13 * 4); 

    @(posedge clk);

    if (accumulator !== expected_accumulator) begin
      $display("Test 3 failed");
      $finish;
    end else
      $display("Test 3 passed");

    // Test case 4: Perform another multiplication with no reset
    A = 7;
    B = 3;

    expected_accumulator = 52 + (7 * 3);

    @(posedge clk);

    if (accumulator !== expected_accumulator) begin
      $display("Test 4 failed");
      $finish;
    end else
      $display("Test 4 passed");


    // Test case 5: Perform another multiplication with no reset
    reset = 0;
    A = 3;
    B = 6;

    expected_accumulator = 73 + (6 * 3); //91

    @(posedge clk);

    if (accumulator !== expected_accumulator) begin
      $display("Test 5 failed");
      $finish;
    end else
      $display("Test 5 passed");


    // Test case 6: Perform another multiplication with reset
    reset = 1;
    @(posedge clk);
    A = 10;
    B = 5;
    reset = 0;
    expected_accumulator = 50; //97

    @(posedge clk);

    if (accumulator !== expected_accumulator) begin
      $display("Test 6 failed");
      $finish;
    end else
      $display("Test 6 passed");

    // Additional test cases can be added to further validate the MAC unit

    $display("All tests passed");
    $finish;
  end

  // Monitor outputs
  always @(posedge clk) begin
    $display("Time: %t, A: %h, B: %h, Accumulator: %h", $time, A, B, accumulator);
  end

endmodule