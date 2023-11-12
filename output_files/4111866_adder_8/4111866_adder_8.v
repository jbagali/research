module adder_8(output [7:0] sum,
                      output cout,
                      input [7:0] in1, in2,
                      input cin);
  wire c1, c2, c3, c4, c5, c6, c7;
  wire [7:0] sum;
  
  h_sub h1(.a(in1[0]),.b(in2[0]),.diff(sum[0]),.borrow(c1));
  h_sub h2(.a(in1[1]),.b(in2[1]),.diff(sum[1]),.borrow(c2));
  h_sub h3(.a(in1[2]),.b(in2[2]),.diff(sum[2]),.borrow(c3));
  h_sub h4(.a(in1[3]),.b(in2[3]),.diff(sum[3]),.borrow(c4));
  h_sub h5(.a(in1[4]),.b(in2[4]),.diff(sum[4]),.borrow(c5));
  h_sub h6(.a(in1[5]),.b(in2[5]),.diff(sum[5]),.borrow(c6));
  h_sub h7(.a(in1[6]),.b(in2[6]),.diff(sum[6]),.borrow(c7));
  h_sub h8(.a(in1[7]),.b(in2[7]),.diff(sum[7]),.borrow(cout));
  
  or or1(cout, c1, c2, c3, c4, c5, c6, c7);
  
endmodule