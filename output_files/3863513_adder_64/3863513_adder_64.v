module adder_64(output [63:0] sum,
                      output cout,
                      input [63:0] in1, in2,
                      input cin);
                      
    wire [63:0] sum;
    wire cout;
    
    wire [63:0] c;
    
    adder_32 a0(sum[31:0], c[0], in1[31:0], in2[31:0], cin);
    adder_32 a1(sum[63:32], cout, in1[63:32], in2[63:32], c[0]);
    
endmodule