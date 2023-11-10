module top_module(
    input clk,
    input reset,
    input ena,
    output pm,
    output [7:0] hh,
    output [7:0] mm,
    output [7:0] ss); 

    reg [7:0] hh;
    reg [7:0] mm;
    reg [7:0] ss;
    reg pm;

    // Counters for hours, minutes, and seconds
    always @(posedge clk) begin
        if (reset) begin
            hh <= 12;
            mm <= 0;
            ss <= 0;
            pm <= 0;
        end else if (ena) begin
            hh <= hh + 1;
            if (hh == 13) begin
                hh <= 1;
                pm <= 1;
            end
        end
    end

    // Output signals
    assign pm = (hh >= 13);
    assign [7:0] hh = hh;
    assign [7:0] mm = mm;
    assign [7:0] ss = ss;

endmodule