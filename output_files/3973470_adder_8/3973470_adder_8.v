module adder_8(output [7:0] sum,
                      output cout,
                      input [7:0] in1, in2,
                      input cin);


    wire [7:0] sum;
    wire cout;
    wire [7:0] in1;
    wire [7:0] in2;
    wire cin;

    CLA_8 adder(sum, cout, in1, in2, cin);

    assign sum = sum;
    assign cout = cout;
    assign in1 = in1;
    assign in2 = in2;
    assign cin = cin;

endmodule