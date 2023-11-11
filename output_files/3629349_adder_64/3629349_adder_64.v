module adder_64(output [63:0] sum,
                      output cout,
                      input [63:0] in1, in2,
                      input cin);
  wire [63:0] sum;
  wire cout;
  wire [63:0] in1;
  wire [63:0] in2;
  wire cin;
  wire [63:0] carry;
  input [63:0] carry;
  assign carry[63] = cin;
  genvar i;
  generate
    for (i = 0; i < 64; i = i + 1)
      begin : adder_bits
        if (i == 0)
          full_adder fa (sum[i], carry[i], in1[i], in2[i], carry[i]);
        else
          full_adder fa (sum[i], carry[i], sum[i-1], in1[i], carry[i-1]);
      end
    assign cout = carry[64];
  endgenerate
endmodule