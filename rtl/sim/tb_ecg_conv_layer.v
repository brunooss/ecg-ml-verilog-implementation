// tb_ecg_conv_layer.v - Testbench do acelerador de convolucao gerado pelo NNgen.
//
// Papel: emula o "host" (CPU) e a DDR3.
//   1. A RAM AXI4 (axi_ram_model) e pre-carregada com sinal + pesos + bias
//      (imagem gerada por scripts/04_make_sim_data.py).
//   2. O testbench programa o acelerador pelo barramento AXI4-Lite (saxi_*):
//      habilita interrupcao (IER) e escreve 1 no registrador START.
//   3. O acelerador le os dados da RAM em bursts AXI4 (maxi_*), computa a
//      convolucao int16 e escreve o resultado de volta na RAM.
//   4. Ao receber irq (ou BUSY=0), o resultado e comparado palavra a palavra
//      com o golden calculado em NumPy (ng.eval). Imprime RESULT: PASS/FAIL.
//
// Parametros da rodada (arquivo gerado): sim_config.vh
//   `DUT_MODULE, `RAM_INIT_FILE, `GOLDEN_FILE, `OUT_ADDR, `OUT_WORDS16,
//   `MEM_BYTES, `TIMEOUT_CYCLES

`timescale 1ns / 1ps

`include "sim_config.vh"

module tb_ecg_conv_layer;

    // mapa de registradores de controle do NNgen (indice x 4 bytes)
    localparam REG_START = 32'h10;  // (W) escrever 1 dispara a execucao
    localparam REG_BUSY  = 32'h14;  // (R) 1 enquanto executa
    localparam REG_ISR   = 32'h24;  // (R) interrupt status
    localparam REG_IER   = 32'h28;  // (W) interrupt enable
    localparam REG_IAR   = 32'h2c;  // (W) interrupt acknowledge

    // ------------------------------------------------------------------
    // clock / reset (100 MHz)
    // ------------------------------------------------------------------
    reg clk = 1'b0;
    reg resetn = 1'b0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // sinais AXI4 master (DUT -> RAM) e AXI4-Lite slave (TB -> DUT)
    // ------------------------------------------------------------------
    wire        irq;

    wire [31:0] maxi_awaddr;
    wire [7:0]  maxi_awlen;
    wire        maxi_awvalid, maxi_awready;
    wire [127:0] maxi_wdata;
    wire [15:0] maxi_wstrb;
    wire        maxi_wlast, maxi_wvalid, maxi_wready;
    wire [1:0]  maxi_bresp;
    wire        maxi_bvalid, maxi_bready;
    wire [31:0] maxi_araddr;
    wire [7:0]  maxi_arlen;
    wire        maxi_arvalid, maxi_arready;
    wire [127:0] maxi_rdata;
    wire [1:0]  maxi_rresp;
    wire        maxi_rlast, maxi_rvalid, maxi_rready;

    reg  [31:0] saxi_awaddr = 0;
    reg         saxi_awvalid = 0;
    wire        saxi_awready;
    reg  [31:0] saxi_wdata = 0;
    reg         saxi_wvalid = 0;
    wire        saxi_wready;
    wire [1:0]  saxi_bresp;
    wire        saxi_bvalid;
    reg         saxi_bready = 0;
    reg  [31:0] saxi_araddr = 0;
    reg         saxi_arvalid = 0;
    wire        saxi_arready;
    wire [31:0] saxi_rdata;
    wire [1:0]  saxi_rresp;
    wire        saxi_rvalid;
    reg         saxi_rready = 0;

    // ------------------------------------------------------------------
    // DUT: acelerador gerado pelo NNgen
    // ------------------------------------------------------------------
    `DUT_MODULE dut (
        .CLK(clk),
        .RESETN(resetn),
        .irq(irq),

        .maxi_awaddr(maxi_awaddr), .maxi_awlen(maxi_awlen),
        .maxi_awsize(), .maxi_awburst(), .maxi_awlock(),
        .maxi_awcache(), .maxi_awprot(), .maxi_awqos(), .maxi_awuser(),
        .maxi_awvalid(maxi_awvalid), .maxi_awready(maxi_awready),
        .maxi_wdata(maxi_wdata), .maxi_wstrb(maxi_wstrb),
        .maxi_wlast(maxi_wlast), .maxi_wvalid(maxi_wvalid),
        .maxi_wready(maxi_wready),
        .maxi_bresp(maxi_bresp), .maxi_bvalid(maxi_bvalid),
        .maxi_bready(maxi_bready),
        .maxi_araddr(maxi_araddr), .maxi_arlen(maxi_arlen),
        .maxi_arsize(), .maxi_arburst(), .maxi_arlock(),
        .maxi_arcache(), .maxi_arprot(), .maxi_arqos(), .maxi_aruser(),
        .maxi_arvalid(maxi_arvalid), .maxi_arready(maxi_arready),
        .maxi_rdata(maxi_rdata), .maxi_rresp(maxi_rresp),
        .maxi_rlast(maxi_rlast), .maxi_rvalid(maxi_rvalid),
        .maxi_rready(maxi_rready),

        .saxi_awaddr(saxi_awaddr), .saxi_awcache(4'h3), .saxi_awprot(3'h0),
        .saxi_awvalid(saxi_awvalid), .saxi_awready(saxi_awready),
        .saxi_wdata(saxi_wdata), .saxi_wstrb(4'hf),
        .saxi_wvalid(saxi_wvalid), .saxi_wready(saxi_wready),
        .saxi_bresp(saxi_bresp), .saxi_bvalid(saxi_bvalid),
        .saxi_bready(saxi_bready),
        .saxi_araddr(saxi_araddr), .saxi_arcache(4'h3), .saxi_arprot(3'h0),
        .saxi_arvalid(saxi_arvalid), .saxi_arready(saxi_arready),
        .saxi_rdata(saxi_rdata), .saxi_rresp(saxi_rresp),
        .saxi_rvalid(saxi_rvalid), .saxi_rready(saxi_rready)
    );

    // ------------------------------------------------------------------
    // RAM AXI4 (DDR3 simulada), pre-carregada com sinal + pesos
    // ------------------------------------------------------------------
    axi_ram_model #(
        .DATA_WIDTH(128),
        .ADDR_WIDTH(32),
        .MEM_BYTES(`MEM_BYTES),
        .INIT_FILE(`RAM_INIT_FILE)
    ) u_ram (
        .clk(clk), .resetn(resetn),
        .s_awaddr(maxi_awaddr), .s_awlen(maxi_awlen),
        .s_awvalid(maxi_awvalid), .s_awready(maxi_awready),
        .s_wdata(maxi_wdata), .s_wstrb(maxi_wstrb), .s_wlast(maxi_wlast),
        .s_wvalid(maxi_wvalid), .s_wready(maxi_wready),
        .s_bresp(maxi_bresp), .s_bvalid(maxi_bvalid), .s_bready(maxi_bready),
        .s_araddr(maxi_araddr), .s_arlen(maxi_arlen),
        .s_arvalid(maxi_arvalid), .s_arready(maxi_arready),
        .s_rdata(maxi_rdata), .s_rresp(maxi_rresp), .s_rlast(maxi_rlast),
        .s_rvalid(maxi_rvalid), .s_rready(maxi_rready)
    );

    // ------------------------------------------------------------------
    // tasks de master AXI4-Lite
    // ------------------------------------------------------------------
    task axil_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            saxi_awaddr  <= addr;
            saxi_awvalid <= 1'b1;
            saxi_wdata   <= data;
            saxi_wvalid  <= 1'b1;
            saxi_bready  <= 1'b1;
            // aguarda handshakes de AW e W (podem ocorrer em ciclos distintos)
            fork
                begin
                    wait (saxi_awvalid && saxi_awready);
                    @(posedge clk);
                    saxi_awvalid <= 1'b0;
                end
                begin
                    wait (saxi_wvalid && saxi_wready);
                    @(posedge clk);
                    saxi_wvalid <= 1'b0;
                end
            join
            wait (saxi_bvalid);
            @(posedge clk);
            saxi_bready <= 1'b0;
        end
    endtask

    task axil_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            saxi_araddr  <= addr;
            saxi_arvalid <= 1'b1;
            saxi_rready  <= 1'b1;
            wait (saxi_arvalid && saxi_arready);
            @(posedge clk);
            saxi_arvalid <= 1'b0;
            wait (saxi_rvalid);
            data = saxi_rdata;
            @(posedge clk);
            saxi_rready <= 1'b0;
        end
    endtask

    // ------------------------------------------------------------------
    // golden de referencia (NumPy / ng.eval)
    // ------------------------------------------------------------------
    reg [15:0] golden [0:`OUT_WORDS16-1];

    // ------------------------------------------------------------------
    // sequencia principal
    // ------------------------------------------------------------------
    integer i, errors;
    reg [31:0] rd;
    reg [15:0] got;
    real t_start, t_end;

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("tb_ecg_conv_layer.vcd");
            $dumpvars(1, tb_ecg_conv_layer);
        end

        $readmemh(`GOLDEN_FILE, golden);

        // reset
        resetn = 1'b0;
        repeat (20) @(posedge clk);
        resetn = 1'b1;
        repeat (10) @(posedge clk);

        $display("[tb] habilitando interrupcao (IER) e disparando START...");
        axil_write(REG_IER, 32'h3);
        t_start = $realtime;
        axil_write(REG_START, 32'h1);

        // espera o fim: irq OU busy=0 (poll)
        rd = 32'h1;
        while (rd != 0 && !irq) begin
            repeat (200) @(posedge clk);
            axil_read(REG_BUSY, rd);
        end
        wait (irq === 1'b1 || rd == 0);
        t_end = $realtime;
        $display("[tb] fim da execucao: irq=%b busy=%0d (%.1f us, ~%0d ciclos)",
                 irq, rd, (t_end - t_start) / 1000.0,
                 $rtoi((t_end - t_start) / 10.0));

        // reconhece a interrupcao
        axil_read(REG_ISR, rd);
        $display("[tb] ISR = 0x%08x", rd);
        axil_write(REG_IAR, rd);

        // despeja a saida e compara com o golden
        u_ram.dump16(`OUT_ADDR, `OUT_WORDS16, "output.hex");

        errors = 0;
        for (i = 0; i < `OUT_WORDS16; i = i + 1) begin
            got = u_ram.get16(`OUT_ADDR + 2*i);
            if (got !== golden[i]) begin
                if (errors < 20)
                    $display("[tb] MISMATCH @%0d: dut=%04x golden=%04x",
                             i, got, golden[i]);
                errors = errors + 1;
            end
        end

        if (errors == 0) begin
            $display("[tb] %0d palavras conferidas contra o golden.", `OUT_WORDS16);
            $display("RESULT: PASS");
        end else begin
            $display("[tb] %0d/%0d palavras divergentes.", errors, `OUT_WORDS16);
            $display("RESULT: FAIL");
        end
        $finish;
    end

    // watchdog
    initial begin
        repeat (`TIMEOUT_CYCLES) @(posedge clk);
        $display("[tb] TIMEOUT apos %0d ciclos", `TIMEOUT_CYCLES);
        $display("RESULT: FAIL");
        $finish;
    end

endmodule
