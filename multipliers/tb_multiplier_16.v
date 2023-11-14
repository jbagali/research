`timescale 1 ns/10 ps

module tb_multiplier_16;

  reg [15:0] A, B;
  wire [31:0] product;

  // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns
  localparam period = 2;

  // Instantiate the 8-bit unsigned multiplier
  multiplier_16 uut(
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
    A = 16'b0000000000000000;
    B = 16'b0000000000000000;
    #period;
    if (product !== 32'b00000000000000000000000000000000) begin
      $display("Test 1 failed");
      $finish;
    end else
      $display("Test 1 passed");

    // Test case 2: 5 * 3 = 15
    A = 16'b0000000000000101;
    B = 16'b0000000000000011;
    #period;
    if (product !== 32'b00000000000000000000000000001111) begin
      $display("Test 2 failed");
      $finish;
    end else
      $display("Test 2 passed");

    // Test case 3: Multiplying by zero (any value * 0 = 0)
    A = 16'b1001010011011010;
    B = 16'b0000000000000000;
    #period;
    if (product !== 32'b00000000000000000000000000000000) begin
      $display("Test 3 failed");
      $finish;
    end else
      $display("Test 3 passed");

    // Test case 4: Multiplying by one (any value * 1 = same value)
    A = 16'b1010101001011010;
    B = 16'b0000000000000001;
    #period;
    if (product !== 32'b00000000000000001010101001011010) begin
      $display("Test 4 failed");
      $finish;
    end else
      $display("Test 4 passed");

    // Test case 5: Multiplying by 255 (any value * 255 = value << 8)
    A = 16'b0000000001111001;
    B = 16'b0000000011111111;
    #period;
    if (product !== 32'b00000000000000000111100010000111) begin
      $display("Test 5 failed");
      $finish;
    end else
      $display("Test 5 passed");

    // Test case 6: Multiplying by 2 (any value * 2 = value << 1)
    A = 16'b0000000001010101;
    B = 16'b0000000000000010;
    #period;
    if (product !== 32'b00000000000000000000000010101010) begin
      $display("Test 6 failed");
      $finish;
    end else
      $display("Test 6 passed");

    // Test case 7: Multiplying 51733 by 13978 (any value * 2 = value << 1)
    A = 16'b1100101000010101;
    B = 16'b0011011010011010;
    #period;
    if (product !== 32'b00101011000110011111111010100010) begin
      $display("Test 6 failed");
      $finish;
    end else
      $display("Test 7 passed");

    // Test case 8: Multiplying binary values 52 and 31
    A = 16'b0000000000110100; // Binary representation of 52
    B = 16'b0000000000011111; // Binary representation of 31
    #period;
    if (product !== 32'b00000000000000000000011001001100) begin
      $display("Test 8 failed");
      $finish;
    end else
      $display("Test 8 passed");
    
    $display("all tests passed");
    $finish;

  end
  // Monitor outputs
  always @(posedge clk) begin
    $display("Time: %t, A: %h, B: %h, Product: %h", $time, A, B, product);
  end

endmodule