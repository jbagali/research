`timescale 1 ns/1 ps

module tb_mac_16;

  reg clk, reset;
  reg [15:0] A, B; // Assuming N=4 for simplicity
  reg [31:0] expected_accumulator;
  wire [31:0] accumulator;

  // Instantiate the MACUnit module
  mac_16 uut (
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
    A = 0;
    B = 0;
    #15 reset = 0;

    // Wait for a few clock cycles
    #15;
    @(posedge clk);
    // Check if the accumulator is in the initial state
    if (accumulator !== 0) begin
      $display("Test 1 failed");
      $finish;
    end else
      $display("Test 1 passed");

    // Test case 2: Perform a multiplication and check the accumulator
    A = 2; // 2
    B = 3; // 3
    

    expected_accumulator = 6; // 6
    @(posedge clk);
    if (accumulator !== expected_accumulator) begin
      $display("Test 2 failed");
      $finish;
    end else
      $display("Test 2 passed");

    // Test case 3: Perform another multiplication with reset
    reset = 1;
    @(posedge clk);
    A = 13; // 13
    B = 4; // 4
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

// Test case 7: Perform another multiplication with reset
    reset = 1;
    @(posedge clk);
    A = 201;
    B = 130;
    reset = 0;
    expected_accumulator = 201*130; //26130

    @(posedge clk);

    if (accumulator !== expected_accumulator) begin
      $display("Test 7 failed");
      $finish;
    end else
      $display("Test 7 passed");

// Test case 8: Perform another multiplication with reset
    A = 14;
    B = 2;
    reset = 0;
    expected_accumulator = 26158; //26130

    @(posedge clk);

    if (accumulator !== expected_accumulator) begin
      $display("Test 8 failed");
      $finish;
    end else
      $display("Test 8 passed");

// Test case 9: Perform reset
    A = 0;
    B = 0;
    reset = 1;
    expected_accumulator = 0; //26130

    @(posedge clk);

    if (accumulator !== expected_accumulator) begin
      $display("Test 9 failed");
      $finish;
    end else
      $display("Test 9 passed");

// Test case 10: Perform another multiplication with reset
    A = 32091;
    B = 11232;
    reset = 0;
    expected_accumulator = 360446112; //26130

    @(posedge clk);
    if (accumulator !== expected_accumulator) begin
      $display("Test 10 failed");
      $finish;
    end else
      $display("Test 10 passed");

// Test case 11: Perform another multiplication with reset
    A = 2;
    B = 8;
    reset = 0;
    expected_accumulator = 360446128; //26130

    @(posedge clk);
    if (accumulator !== expected_accumulator) begin
      $display("Test 11 failed");
      $finish;
    end else
      $display("Test 11 passed");

    $display("All tests passed");
    $finish;
  end


  // Monitor outputs
  always @(posedge clk) begin
    $display("Time: %t, A: %h, B: %h, Accumulator: %h", $time, A, B, accumulator);
  end

endmodule