module adder_32(output [31:0] sum,
                      output cout,
                      input [31:0] in1, in2,
                      input cin);

   wire [31:0] sum;
   wire cout;
   wire [31:0] in1;
   wire [31:0] in2;
   wire cin;

   adder_32_bitwise adder(sum, cout, in1, in2, cin);

endmodule