`timescale 1 ns/10 ps

module tb_multiplier_8;

  reg [7:0] A, B;
  wire [15:0] product;

  // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns
  localparam period = 2;

  // Instantiate the 8-bit unsigned multiplier
  multiplier_8 uut(
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
    A = 8'b00000000;
    B = 8'b00000000;
    #period;
    if (product !== 16'b0000000000000000) begin
      $display("Test 1 failed");
      $finish;
    end else
      $display("Test 1 passed");

    // Test case 2: 5 * 3 = 15
    A = 8'b00000101;
    B = 8'b00000011;
    #period;
    if (product !== 16'b0000000000001111) begin
      $display("Test 2 failed");
      $finish;
    end else
      $display("Test 2 passed");

    // Test case 3: Multiplying by zero (any value * 0 = 0)
    A = 8'b11011010;
    B = 8'b00000000;
    #period;
    if (product !== 16'b0000000000000000) begin
      $display("Test 3 failed");
      $finish;
    end else
      $display("Test 3 passed");

    // Test case 4: Multiplying by one (any value * 1 = same value)
    A = 8'b10101010;
    B = 8'b00000001;
    #period;
    if (product !== 16'b0000000010101010) begin
      $display("Test 4 failed");
      $finish;
    end else
      $display("Test 4 passed");

    // Test case 5: Multiplying by 255 (any value * 255 = value << 8)
    A = 8'b01111001;
    B = 8'b11111111;
    #period;
    if (product !== 16'b0111100010000111) begin
      $display("Test 5 failed");
      $finish;
    end else
      $display("Test 5 passed");

    // Test case 6: Multiplying by 2 (any value * 2 = value << 1)
    A = 8'b01010101;
    B = 8'b00000010;
    #period;
    if (product !== 16'b0000000010101010) begin
      $display("Test 6 failed");
      $finish;
    end else
      $display("Test 6 passed");

    // Test case 8: Multiplying binary values 52 and 31
    A = 8'b00110100; // Binary representation of 52
    B = 8'b00011111; // Binary representation of 31
    #period;
    if (product !== 16'b0000011001001100) begin
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