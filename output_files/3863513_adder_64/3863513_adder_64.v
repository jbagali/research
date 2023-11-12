module adder_64(output [63:0] sum,
                      output cout,
                      input [63:0] in1, in2,
                      input cin);

assign cin = 1'b0;

generate

genvar i;

for (i = 0; i < 64; i = i + 1)
begin: adder_loop

wire c1, c2, c3;

adder_1bit a1(sum[i], c1, in1[i], in2[i]);
adder_1bit a2(sum[i+1], c2, in1[i], in2[i+1]);
adder_1bit a3(sum[i+2], c3, in1[i+1], in2[i+2]);
adder_1bit a4(sum[i+3], cout, in1[i+2], in2[i+3]);

assign c1 = cout;
assign c2 = c1;
assign c3 = c2;

end

endgenerate

endmodule