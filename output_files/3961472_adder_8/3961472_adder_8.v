module adder_8(output [7:0] sum,
                      output cout,
                      input [7:0] in1, in2,
                      input cin);

    wire [7:0] sum;
    wire cout;
    wire [7:0] in1;
    wire [7:0] in2;
    wire cin;

    eight_bit_adder adder(sum, cout, in1, in2, cin);

endmodule