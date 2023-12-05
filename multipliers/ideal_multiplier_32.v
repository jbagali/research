module multiplier_32 (
    output [63:0] product,
    input [31:0] A,
    input [31:0] B
);

  wire [31:0] p0, p1, p2, p3;

  multiplier_16 uut0 (.product(p0), .A(A[15:0]), .B(B[15:0]));
  multiplier_16 uut1 (.product(p1), .A(A[31:16]), .B(B[31:16]));
  multiplier_16 uut2 (.product(p2), .A(A[47:32]), .B(B[47:32]));
  multiplier_16 uut3 (.product(p3), .A(A[63:48]), .B(B[63:48]));

  assign product = {p3, p2, p1, p0}; // Concatenate the 32-bit products

endmodule