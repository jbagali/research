module adder_64(output [63:0] sum,
                      output cout,
                      input [63:0] in1, in2,
                      input cin);

    wire [63:0] sum;
    wire cout;
    wire [63:0] in1;
    wire [63:0] in2;
    wire cin;

    // Create a 64-bit carry select line
    wire [64:0] carry_select;

    // Create two 64-bit carry adders
    wire [63:0] carry_in1;
    wire [63:0] carry_in2;
    wire [63:0] carry_out1;
    wire [63:0] carry_out2;

    // Create two 64-bit carry select lines
    wire [64:0] carry_select1;
    wire [64:0] carry_select2;

    // Create two 64-bit carry adders
    wire [63:0] carry_in1_1;
    wire [63:0] carry_in1_2;
    wire [63:0] carry_in2_1;
    wire [63:0] carry_in2_2;
    wire [63:0] carry_out1_1;
    wire [63:0] carry_out1_2;
    wire [63:0] carry_out2_1;
    wire [63:0] carry_out2_2;

    // Create two 64-bit carry select lines
    wire [64:0] carry_select1_1;
    wire [64:0] carry_select1_2;
    wire [64:0] carry_select2_1;
    wire [64:0] carry_select2_2;

    // Create two 64-bit carry adders
    wire [63:0] carry_in1_3;
    wire [63:0] carry_in1_4;
    wire [63:0] carry_in2_3;
    wire [63:0] carry_in2_4;
    wire [63:0] carry_out1_3;
    wire [63:0] carry_out1_4;
    wire [63:0] carry_out2_3;
    wire [63:0] carry_out2_4;

