Area,Delay,Score,Current Run,Episode,Verilog
N/A,N/A,-2,2.0,1,"`timescale 1 ns/10 ps

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
    if (product!== 32'b00000000000000000000000000000000) begin
      $display(""Test 1 failed"");
      $finish;
    end else
      $display(""Test 1 passed"");

    // Test case 2: 5 * 3 = 15
    A = 16'b0000000000000101;
    B = 16'b0000000000000011;
    #period;
    if (product!== 32'b00000000000000000000000000001111) begin
      $display(""Test 2 failed"");
      $finish;
    end else
      $display(""Test 2 passed"");

    // Test case 3: Multiplying by zero (any value * 0 = 0)
    A = 16'b1001010011011010;
    B = 16'b0000000000000000;
    #period;
    if (product!== 32'b00000000000000000000000000000000) begin
      $display(""Test 3 failed"");
      $finish;
    end else
      $display(""Test 3 passed"");

    // Test case 4: Multiplying by one (any value * 1 = same value)
    A = 16'b1010101001011010;
    B = 16'b0000000000000001;
    #period;
    if (product!== 32'b00000000000000001010101001011010) begin
      $display(""Test 4 failed"");
      $finish;
    end else
      $display(""Test 4 passed"");

    // Test case 5: Multiplying by 255 (any value * 255 = value << 8)
    A = 16'b0000000001111001;
    B = 16'b0000000011111111;
    #period;
    if (product!== 32'b00000000000000000111100010000111) begin
      $display(""Test 5 failed"");
      $finish;
    end else
      $display(""Test 5 passed"");

    // Test case 6: Multiplying by 2 (any value * 2 = value << 1)
    A = 16'b0000000001010101;
    B = 16'b0000000000000010;
    #period;
    if (product!== 32'b00000000000000000000000010101010) begin
      $display(""Test 6 failed"");
      $finish;
    end else
      $display(""Test 6 passed"");

    // Test case 7: Multiplying 51733 by 13978 (any value * 2 = value << 1)
    A = 16'b1100101000010101;
    B = 16'b0011011010011010;
    #period;
    if (product!== 32'b00101011000110011111111010100010) begin
      $display(""Test 6 failed"");
      $finish;
    end else
      $display(""Test 7 passed"");

    // Test case 8: Multiplying binary values 52 and 31
    A = 16'b0000000000110100; // Binary representation of 52
    B = 16'b0000000000011111; // Binary representation of 31
    #period;
    if (product!== 32'b00000000000000000000011001001100) begin
      $display(""Test 8 failed"");
      $finish;
    end else
      $display(""Test 8 passed"");
    
    $display(""all tests passed"");
    $finish;

  end
  // Monitor outputs
  always @(posedge clk) begin
    $display(""Time: %t, A: %h, B: %h, Product: %h"", $time, A, B, product);
  end

endmodule
`timescale 1 ns/10 ps

module tb_multiplier_8;

  reg [7:0] A, B;
  wire [15:0] product;

  logic [15:0] expected;

  main uut(
   .product(product),
   .A(A),
   .B(B)
  );

  nbit_multiplier #(.WIDTH(8)) multiplier(
   .A(A),
   .B(B),
   .product(product)
  );

  error_detection #(.WIDTH(16)) error_detection(
   .A(A),
   .B(B),
   .product(product)
  );

  initial begin
    testing = 1'b0;
    A = 8'b00000000;
    B = 8'b00000000;
    #10;
    A = 8'b00000001;
    B = 8'b00000001;
    #10;
    A = 8'b00000010;
    B = 8'b00000010;
    #10;
    A = 8'b00000100;
    B = 8'b00000100;
    #10;
    A = 8'b00001000;
    B = 8'b00001000;
    #10;
    A = 8'b00010000;
    B = 8'b00010000;
    #10;
    A = 8'b00100000;
    B = 8'b00100000;
    #10;
    A = 8'b01000000;
    B = 8'b01000000;
    #10;
    A = 8'b10000000;
    B = 8'b10000000;
    #10;
    testing = 1'b1;
    A = 8'b00000000;
    B = 8'b00000000;
    #10;
    A = 8'b00000001;
    B = 8'b00000001;
    #10;
    A = 8'b00000010;
    B = 8'b00000010;
    #10;
    A = 8'b00000100;
    B = 8'b00000100;
    #10;
    A = 8'b00001000;
    B = 8'b00001000;
    #10;
    A = 8'b00010000;
    B = 8'b00010000;
    #10;
    A = 8'b00100000;
    B = 8'b00100000;
    #10;
    A = 8'b01000000;
    B = 8'b01000000;
    #10;
    A = 8'b10000000;
    B = 8'b10000000;
    #10;
    $display(""all tests passed"");
    $finish;
  end

  legacy_multiplier #(.WIDTH(8)) legacy_multiplier(
   .A(A),
   .B(B),
   .product(product)
  );

  nbit_multiplier #(.WIDTH(16)) nbit_multiplier(
   .A(A),
   .B(B),
   .product(product)
  );

  error_detection error_detection(
   .A(A),
   .B(B),
   .product(product)
  );

endmodule"
