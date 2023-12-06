`timescale 1 ns/10 ps

module tb_multiplier_32;

  reg [31:0] A, B;
  wire [63:0] product;

  // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns
  localparam period = 2;

  // Instantiate the 8-bit unsigned multiplier
  multiplier_32 uut(
    .product(product),
    .A(A),
    .B(B)
  );

  // Clock generation
  reg clk;
  always #5 clk = ~clk;

  // Test cases
  initial begin
    // Test case 1: 0 * 0 = 0
    A = 0;
    B = 0;
    #period;
    if (product !== 0) begin
      $display("Test 1 failed");
      $finish;
    end else
      $display("Test 1 passed");

    // Test case 2: 5 * 3 = 15
    A = 5;
    B = 3;
    #period;
    if (product !== 15) begin
      $display("Test 2 failed");
      $finish;
    end else
      $display("Test 2 passed");

    // Test case 3: Multiplying by zero (any value * 0 = 0)
    A = 40221;
    B = 0;
    #period;
    if (product !== 0) begin
      $display("Test 3 failed");
      $finish;
    end else
      $display("Test 3 passed");

    // Test case 4: Multiplying by one (any value * 1 = same value)
    A = 43610;
    B = 1;
    #period;
    if (product !== 43610) begin
      $display("Test 4 failed");
      $finish;
    end else
      $display("Test 4 passed");

    // Test case 5: Multiplying by 255 (any value * 255 = value << 8)
    A = 121;
    B = 255;
    #period;
    if (product !== 30855) begin
      $display("Test 5 failed");
      $finish;
    end else
      $display("Test 5 passed");

    // Test case 6: Multiplying by 2 (any value * 2 = value << 1)
    A = 85;
    B = 2;
    #period;
    if (product !== 170) begin
      $display("Test 6 failed");
      $finish;
    end else
      $display("Test 6 passed");

    // Test case 7: Multiplying 51733 by 13978 (any value * 2 = value << 1)
    A = 51733;
    B = 13978;
    #period;
    if (product !== 723123874) begin
      $display("Test 7 failed");
      $finish;
    end else
      $display("Test 7 passed");

    // Test case 8: Multiplying binary values 52 and 31
    A = 52; // Binary representation of 52
    B = 31; // Binary representation of 31
    #period;
    if (product !== 1612) begin
      $display("Test 8 failed");
      $finish;
    end else
      $display("Test 8 passed");
    

    // Test case 9: Multiplying binary values 52 and 31
    A = 1923842001; // Binary representation of 52
    B = 1409280110; // Binary representation of 31
    #period;
    if (product !== 2711232266791900110) begin
      $display("Test 9 failed");
      $finish;
    end else
      $display("Test 9 passed");
    
    $display("all tests passed");
    $finish;


  end
  // Monitor outputs
  always @(posedge clk) begin
    $display("Time: %t, A: %h, B: %h, Product: %h", $time, A, B, product);
  end

endmodule