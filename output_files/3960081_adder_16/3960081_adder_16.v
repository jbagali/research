module adder_16(output [15:0] sum,
                      output cout,
                      input [15:0] in1, in2,
                      input cin);
  wire c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
  wire [15:0] sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15;
  
  MAOI222_X1 U1 (sum1, c1, in1[15], in2[15], cin);
  MAOI222_X1 U2 (sum2, c2, in1[14], in2[14], c1);
  MAOI222_X1 U3 (sum3, c3, in1[13], in2[13], c2);
  MAOI222_X1 U4 (sum4, c4, in1[12], in2[12], c3);
  MAOI222_X1 U5 (sum5, c5, in1[11], in2[11], c4);
  MAOI222_X1 U6 (sum6, c6, in1[10], in2[10], c5);
  MAOI222_X1 U7 (sum7, c7, in1[9], in2[9], c6);
  MAOI222_X1 U8 (sum8, c8, in1[8], in2[8], c7);
  MAOI222_X1 U9 (sum9, c9, in1[7], in2[7], c8);
  MAOI222_X1 U10 (sum10, c10, in1[6], in2[6], c9);
  MAOI222_X1 U11 (sum11, c11, in1[5], in2[5], c10);
  MAOI222_X1 U12 (sum12, c12, in1[4], in2[4], c11);
  MAOI222_X1 U13 (sum13, c13, in1[3], in2[3], c12);
  MAOI222_X1 U14 (sum14, c14, in1[2], in2[2], c13);
  MAOI222_X1 U15 (sum15, c15, in1[1], in2[1], c14);
  
  MAOI222_X1 U16 (sum, cout, in1[0], in2[0], c15);
  
endmodule