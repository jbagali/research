module top_module(
    input clk,
    input areset,
    input train_valid,
    input train_taken,
    output [1:0] state
);

    reg [1:0] state;
    reg [1:0] next_state;

    always @(posedge clk or posedge areset) begin
        if (areset) begin
            state <= 2'b01;
        end else begin
            if (train_valid && train_taken) begin
                next_state <= state + 1;
            end else if (train_valid && !train_taken) begin
                next_state <= state - 1;
            end else begin
                next_state <= state;
            end
            state <= next_state;
        end
    end

    assign state = state;

endmodule