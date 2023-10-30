module adder(output [3:0] sum, output cout, input [3:0] in1, in2, input cin);

wire c1, c2, c3;!

full_adder FA1(sum[0], c1, in1[0], in2[0], cin);
full_adder FA2(sum[1], c2, in1[1], in2[1], c1);
full_adder FA3(sum[2], c3, in1[2], in2[2], c2);
full_adder FA4(sum[3], cout, in1[3], in2[3], c3);

endmodule

module full_adder(output sum, output cout, input in1, in2, cin);

wire x, y, z;

xor(sum, in1, in2, cin);
and(x, in1, in2);
and(y, in1, cin);
and(z, in2, cin);
or(cout, x, y, z);

endmodule

module test_adder;

wire [3:0] sum;
wire cout;
reg [3:0] in1, in2;
reg cin;

adder a1(sum, cout, in1, in2, cin);

initial
begin
    $monitor("in1 = %b, in2 = %b, cin = %b, sum = %b, cout = %b", in1, in2, cin, sum, cout);
    in1 = 4'b0000; in2 = 4'b0000; cin = 1'b0;
    #10 in1 = 4'b0001; in2 = 4'b0001;
    #10 in1 = 4'b0010; in2 = 4'b0010;
    #10 in1 = 4'b0011; in2 = 4'b0011;
    #10 in1 = 4'b0100; in2 = 4'b0100;
    #10 in1 = 4'b0101; in2 = 4'b0101;
    #10 in1 = 4'b0110; in2 = 4'b0110;
    #10 in1 = 4'b0111; in2 = 4