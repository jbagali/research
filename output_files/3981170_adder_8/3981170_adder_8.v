module adder_8(output [7:0] sum,
                      output cout,
                      input [7:0] in1, in2,
                      input cin);
  wire c1,c2,c3,c4,c5,c6,c7;
  full_adder_1 FA1(sum[0], c1, in1[0], in2[0], cin);
  full_adder_1 FA2(sum[1], c2, in1[1], in2[1], c1);
  full_adder_1 FA3(sum[2], c3, in1[2], in2[2], c2);
  full_adder_1 FA4(sum[3], c4, in1[3], in2[3], c3);
  full_adder_1 FA5(sum[4], c5, in1[4], in2[4], c4);
  full_adder_1 FA6(sum[5], c6, in1[5], in2[5], c5);
  full_adder_1 FA7(sum[6], c7, in1[6], in2[6], c6);
  full_adder_1 FA8(sum[7], cout, in1[7], in2[7], c7);
endmodule

module full_adder_1(output wire sum,
                     output wire cout,
                     input wire in1, in2, cin);
  wire t1, t2, t3;
  xor(t1, in1, in2);
  xor(sum, t1, cin);
  and(t2, t1, cin);
  and(t3, in1, in2);
  or(cout, t2, t3);
endmodule