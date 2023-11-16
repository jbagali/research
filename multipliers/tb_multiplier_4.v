`timescale 1 ns/10 ps

module tb_multiplier_4;

  reg [3:0] A, B;
  wire [7:0] product;

  // duration for each bit = 2 * timescale = 2 * 1 ns  = 2ns
  localparam period = 2;

  // Instantiate the 8-bit unsigned multiplier
  multiplier_4 uut(
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
    A = 4'b0000;
    B = 4'b0000;
    #period;
    if (product !== 8'b00000000) begin
      $display("Test 1 failed");
      $finish;
    end else
      $display("Test 1 passed");

    // Test case 2: 5 * 3 = 15
    A = 4'b0101;
    B = 4'b0011;
    #period;
    if (product !== 8'b00001111) begin
      $display("Test 2 failed");
      $finish;
    end else
      $display("Test 2 passed");

    // Test case 3: Multiplying by zero (any value * 0 = 0)
    A = 4'b1110;
    B = 4'b00000000;
    #period;
    if (product !== 8'b00000000) begin
      $display("Test 3 failed");
      $finish;
    end else
      $display("Test 3 passed");

    // Test case 4: Multiplying by one (any value * 1 = same value)
    A = 4'b1011;
    B = 4'b0001;
    #period;
    if (product !== 8'b00001011) begin
      $display("Test 4 failed");
      $finish;
    end else
      $display("Test 4 passed");

    // Test case 5: Multiplying by 255 (any value * 255 = value << 8)
    A = 4'b0110;
    B = 4'b1111;
    #period;
    if (product !== 8'b01011010) begin
      $display("Test 5 failed");
      $finish;
    end else
      $display("Test 5 passed");

    // Test case 6: Multiplying by 2 (any value * 2 = value << 1)
    A = 4'b1001;
    B = 4'b0010;
    #period;
    if (product !== 8'b00010010) begin
      $display("Test 6 failed");
      $finish;
    end else
      $display("Test 6 passed");

    // Test case 8: Multiplying binary values 13 and 10
    A = 4'b1101; // Binary representation of 13
    B = 4'b1010; // Binary representation of 10
    #period;
    if (product !== 8'b10000010) begin
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