module adder_64(output [63:0] sum,
                      output cout,
                      input [63:0] in1, in2,
                      input cin);
  wire c1, c2, c3;
  full_adder FA1(sum[0], c1, in1[0], in2[0], cin);
  full_adder FA2(sum[1], c2, in1[1], in2[1], c1);
  full_adder FA3(sum[2], c3, in1[2], in2[2], c2);
  full_adder FA4(sum[3], cout, in1[3], in2[3], c3);
endmodule