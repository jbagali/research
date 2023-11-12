module adder_32(output [31:0] sum,
                      output cout,
                      input [31:0] in1, in2,
                      input cin);

    wire c1, c2, c3;
    wire [31:0] sum1, sum2, sum3;
    wire cout1, cout2, cout3;

    adder_32_bit a1(sum1, c1, in1, in2);
    adder_32_bit a2(sum2, c2, in1, in2);
    adder_32_bit a3(sum3, c3, in1, in2);

    adder_32_bit a4(sum, cout, sum1, c1);
    adder_32_bit a5(sum, cout1, sum2, c2);
    adder_32_bit a6(sum, cout2, sum3, c3);

endmodule

module adder_32_bit(output [31:0] sum,
                      output cout,
                      input [31:0] in1, in2);

    wire c1, c2, c3;
    wire [31:0] sum1, sum2, sum3;
    wire cout1, cout2, cout3;

    adder_32_bit_bitwise a1(sum1, c1, in1, in2);
    adder_32_bit_bitwise a2(sum2, c2, in1, in2);
    adder_32_bit_bitwise a3(sum3, c3, in1, in2);

    adder_32_bit_bitwise a4(sum, cout, sum1, c1);
    adder_32_bit_bitwise a5(sum, cout1, sum2, c2);
    adder_32_bit_bitwise a6(sum, cout2, sum3, c3);

endmodule

module adder_32_bit_bitwise(output [31:0] sum,
                      output cout,
                      input [31:0] in1, in2);

    wire c1, c2, c3;
    wire [31:0] sum1, sum2, sum3;
    wire cout1, cout2, cout3;

    adder_32_bit_bitwise_1 a1(sum1, c1, in1, in2);
    adder_32_bit_bitwise_1 a2(sum2, c2, in1, in2);
    adder_32_bit_bitwise_1 a3(sum3, c3, in1, in2);

    adder_32_bit_bitwise_2 a4(sum, cout, sum1, c1);
    adder_32_bit_bitwise_2 a5(sum, cout1, sum2, c2);
    adder_32_bit_bitwise_2 a6(sum, cout2, sum3, c3);

endmodule

module adder_32_bit_bitwise_1(output [31:0] sum,
                      output cout,
                      input [31:0] in1, in2);

    wire c1, c2, c3;
    wire [31:0] sum1, sum2, sum3;
    wire cout1, cout2, cout3;

    adder_32_bit_bitwise_1_bitwise_1 a1(sum1, c1, in1, in2);
    adder_32_bit_bitwise_1_bitwise_1 a2(sum2, c2, in1, in2);
    adder_32_bit_bitwise_1_bitwise_1 a3(sum3, c3, in1, in2);

    adder_32_bit_bitwise_1_bitwise_1 a4(sum, cout, sum1, c1);
    adder_32_bit_bitwise_1_bitwise_1 a5(sum, cout1, sum2, c2);
    adder_32_bit_bitwise_1_bitwise_1 a6(sum, cout2, sum3, c3);

endmodule

module adder_32_bit_bitwise_1_bitwise_1(output [31:0] sum,
                      output cout,
                      input [31:0] in1, in2);

    wire c1, c2, c3;
    wire [31:0] sum1, sum2, sum3;
    wire cout1