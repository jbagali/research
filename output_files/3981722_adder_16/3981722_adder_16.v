module adder_16(output [15:0] sum,
                      output cout,
                      input [15:0] in1, in2,
                      input cin);

    assign {cout, sum} = in1 + in2 + cin;
endmodule