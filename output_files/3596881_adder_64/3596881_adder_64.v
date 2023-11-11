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
  
  full_adder FA0(sum[0], carry[0], in1[0], in2[0], cin);
  full_adder FA1(sum[1], carry[1], in1[1], in2[1], carry[0]);
  full_adder FA2(sum[2], carry[2], in1[2], in2[2], carry[1]);
  full_adder FA3(sum[3], carry[3], in1[3], in2[3], carry[2]);
  full_adder FA4(sum[4], carry[4], in1[4], in2[4], carry[3]);
  full_adder FA5(sum[5], carry[5], in1[5], in2[5], carry[4]);
  full_adder FA6(sum[6], carry[6], in1[6], in2[6], carry[5]);
  full_adder FA7(sum[7], carry[7], in1[7], in2[7], carry[6]);
  full_adder FA8(sum[8], carry[8], in1[8], in2[8], carry[7]);
  full_adder FA9(sum[9], carry[9], in1[9], in2[9], carry[8]);
  full_adder FA10(sum[10], carry[10], in1[10], in2[10], carry[9]);
  full_adder FA11(sum[11], carry[11], in1[11], in2[11], carry[10]);
  full_adder FA12(sum[12], carry[12], in1[12], in2[12], carry[11]);
  full_adder FA13(sum[13], carry[13], in1[13], in2[13], carry[12]);
  full_adder FA14(sum[14], carry[14], in1[14], in2[14],