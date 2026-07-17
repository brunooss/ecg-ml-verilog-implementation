// axi_ram_model.v - Modelo de RAM com interface AXI4 slave (simulacao apenas).
//
// Faz o papel da DDR3 externa na simulacao: o acelerador gerado pelo NNgen
// (ecg_conv_layer*) le o sinal de ECG e os pesos daqui via bursts AXI4 e
// escreve o resultado de volta. A imagem inicial da memoria e carregada de um
// arquivo hex ($readmemh) com uma palavra de 128 bits por linha.
//
// Simplificacoes assumidas (compativeis com o master gerado pelo NNgen):
//   - bursts INCR, awsize/arsize = largura total do barramento (16 bytes);
//   - enderecos alinhados a 16 bytes;
//   - uma transacao pendente por canal (leitura e escrita independentes).
//
// Nao e sintetizavel; uso exclusivo em testbench.

`timescale 1ns / 1ps

module axi_ram_model #(
    parameter DATA_WIDTH = 128,
    parameter ADDR_WIDTH = 32,
    parameter MEM_BYTES  = 65536,          // tamanho da RAM em bytes
    parameter INIT_FILE  = ""              // imagem inicial (hex, 128b/linha)
) (
    input                       clk,
    input                       resetn,

    // canal de endereco de escrita
    input  [ADDR_WIDTH-1:0]     s_awaddr,
    input  [7:0]                s_awlen,
    input                       s_awvalid,
    output reg                  s_awready,

    // canal de dados de escrita
    input  [DATA_WIDTH-1:0]     s_wdata,
    input  [DATA_WIDTH/8-1:0]   s_wstrb,
    input                       s_wlast,
    input                       s_wvalid,
    output reg                  s_wready,

    // canal de resposta de escrita
    output [1:0]                s_bresp,
    output reg                  s_bvalid,
    input                       s_bready,

    // canal de endereco de leitura
    input  [ADDR_WIDTH-1:0]     s_araddr,
    input  [7:0]                s_arlen,
    input                       s_arvalid,
    output reg                  s_arready,

    // canal de dados de leitura
    output reg [DATA_WIDTH-1:0] s_rdata,
    output [1:0]                s_rresp,
    output reg                  s_rlast,
    output reg                  s_rvalid,
    input                       s_rready
);

    localparam BYTES_PER_WORD = DATA_WIDTH / 8;             // 16
    localparam MEM_WORDS      = MEM_BYTES / BYTES_PER_WORD;
    localparam WORD_LSB       = $clog2(BYTES_PER_WORD);     // 4

    reg [DATA_WIDTH-1:0] mem [0:MEM_WORDS-1];

    assign s_bresp = 2'b00;  // OKAY
    assign s_rresp = 2'b00;  // OKAY

    integer k;
    initial begin
        for (k = 0; k < MEM_WORDS; k = k + 1)
            mem[k] = {DATA_WIDTH{1'b0}};
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    // ------------------------------------------------------------------
    // canal de leitura (AR -> R)
    // ------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] raddr;
    reg [8:0]            rbeats;   // beats restantes (arlen+1, ate 256)
    reg                  rbusy;

    always @(posedge clk) begin
        if (!resetn) begin
            s_arready <= 1'b0;
            s_rvalid  <= 1'b0;
            s_rlast   <= 1'b0;
            s_rdata   <= {DATA_WIDTH{1'b0}};
            rbusy     <= 1'b0;
            raddr     <= 0;
            rbeats    <= 0;
        end else begin
            if (!rbusy) begin
                s_arready <= 1'b1;
                if (s_arvalid && s_arready) begin
                    raddr     <= s_araddr;
                    rbeats    <= {1'b0, s_arlen} + 9'd1;
                    rbusy     <= 1'b1;
                    s_arready <= 1'b0;
                end
            end else begin
                if (!s_rvalid || s_rready) begin
                    if (rbeats != 0) begin
                        s_rdata  <= mem[raddr >> WORD_LSB];
                        s_rvalid <= 1'b1;
                        s_rlast  <= (rbeats == 1);
                        raddr    <= raddr + BYTES_PER_WORD;
                        rbeats   <= rbeats - 1;
                    end else begin
                        // ultimo beat aceito
                        s_rvalid <= 1'b0;
                        s_rlast  <= 1'b0;
                        rbusy    <= 1'b0;
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // canal de escrita (AW -> W -> B)
    // ------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] waddr;
    reg                  wbusy;
    integer b;

    always @(posedge clk) begin
        if (!resetn) begin
            s_awready <= 1'b0;
            s_wready  <= 1'b0;
            s_bvalid  <= 1'b0;
            wbusy     <= 1'b0;
            waddr     <= 0;
        end else begin
            if (s_bvalid && s_bready)
                s_bvalid <= 1'b0;

            if (!wbusy) begin
                s_awready <= 1'b1;
                if (s_awvalid && s_awready) begin
                    waddr     <= s_awaddr;
                    wbusy     <= 1'b1;
                    s_awready <= 1'b0;
                    s_wready  <= 1'b1;
                end
            end else begin
                if (s_wvalid && s_wready) begin
                    for (b = 0; b < BYTES_PER_WORD; b = b + 1)
                        if (s_wstrb[b])
                            mem[waddr >> WORD_LSB][8*b +: 8] <= s_wdata[8*b +: 8];
                    waddr <= waddr + BYTES_PER_WORD;
                    if (s_wlast) begin
                        wbusy    <= 1'b0;
                        s_wready <= 1'b0;
                        s_bvalid <= 1'b1;
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // acesso auxiliar para o testbench (leitura de 16 bits por endereco em bytes)
    // ------------------------------------------------------------------
    function [15:0] get16(input [ADDR_WIDTH-1:0] byte_addr);
        reg [DATA_WIDTH-1:0] word;
        begin
            word  = mem[byte_addr >> WORD_LSB];
            get16 = word >> (8 * byte_addr[WORD_LSB-1:0]);
        end
    endfunction

    // despeja uma faixa da memoria em arquivo (uma palavra de 16 bits por linha)
    task dump16(input [ADDR_WIDTH-1:0] byte_addr,
                input integer          nwords,
                input [8*128-1:0]      fname);
        integer fd, i;
        begin
            fd = $fopen(fname, "w");
            for (i = 0; i < nwords; i = i + 1)
                $fdisplay(fd, "%04x", get16(byte_addr + 2*i));
            $fclose(fd);
        end
    endtask

endmodule
