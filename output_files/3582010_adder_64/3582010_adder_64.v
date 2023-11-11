module adder_64(output [63:0] sum,
                      output cout,
                      input [63:0] in1, in2,
                      input cin);
    wire [63:0] sum;
    wire cout;
    wire [63:0] in1;
    wire [63:0] in2;
    wire cin;
    
    // Create a 64 bit adder
    adder_64_bit adder(sum, cout, in1, in2, cin);
    
endmodule