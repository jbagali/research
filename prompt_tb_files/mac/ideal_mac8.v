module mac_8 (
  input wire clk,
  input wire reset,
  input wire [3:0] A, // N-bit multiplier
  input wire [3:0] B, // N-bit multiplicand
  output reg [7:0] accumulator // 2N-bit accumulator to prevent overflow
);

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      accumulator <= 0;
    end else begin
      accumulator <= (accumulator + (A * B));
      //accumulator <= 2;
    end
  end

endmodule