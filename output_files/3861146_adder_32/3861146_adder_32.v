module adder_32(output [31:0] sum,
                      output cout,
                      input [31:0] in1, in2,
                      input cin);

genvar i;

wire [31:0] carry;

generate

for (i = 0; i < 32; i = i + 1)
begin: adder_loop
    wire c1, c2;
    
    if (i == 0)
    begin
        full_adder FA1(sum[i], carry[i], in1[i], in2[i], cin);
        assign c1 = carry[i];
    end
    else
    begin
        full_adder FA1(sum[i], carry[i], in1[i], in2[i], c1);
        assign c2 = carry[i];
    end
end

assign cout = c2;

endgenerate

endmodule

\****************************************************************************])

Computer Architecture Course, Laboratory Sources 
ECE 3700, Laboratory 3700

Professor: Dr. H. El Naga

Lab: ALU

Module: ALU_32

Functionality:

Inputs:

32-bit Inputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit Outputs

Outputs:

32-bit