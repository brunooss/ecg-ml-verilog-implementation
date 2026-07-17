# Simulacao do acelerador NNgen com RAM AXI4 simulada (Icarus Verilog).
#
# Fluxo:  make data  -> gera RTL da instancia de teste + imagem da RAM + golden
#         make run   -> compila (iverilog) e roda (vvp); imprime RESULT: PASS/FAIL
#         make check -> falha (exit != 0) se o resultado nao for PASS
#         make wave  -> roda com dump VCD (tb_ecg_conv_layer.vcd)
#
# Dimensoes da instancia de teste (mesmo operador do modelo, reduzido para a
# simulacao ser rapida): sobrescreva com DIMS, ex.:
#         make check DIMS="--H 128 --Cin 16 --Cout 16"
# (mudar DIMS invalida e regenera dados/RTL automaticamente)

PYTHON   ?= python3
IVERILOG ?= iverilog
VVP      ?= vvp
DIMS     ?=

ROOT       := ..
RTL_SIM    := $(ROOT)/rtl/sim
DUT_RTL    := $(ROOT)/outputs/nngen/ecg_conv_layer_sim.v
BUILD      := build
TB_VVP     := $(BUILD)/tb_ecg_conv_layer.vvp
DIMS_STAMP := $(BUILD)/dims.stamp
DATA_STAMP := data/.stamp

SRCS := $(RTL_SIM)/tb_ecg_conv_layer.v $(RTL_SIM)/axi_ram_model.v $(DUT_RTL)

.PHONY: all data compile run check wave clean force

all: check

# o stamp so muda quando DIMS muda -> forca regeracao dos dados
$(DIMS_STAMP): force
	@mkdir -p $(BUILD)
	@echo '$(DIMS)' | cmp -s - $@ || echo '$(DIMS)' > $@

$(DATA_STAMP): $(DIMS_STAMP) $(ROOT)/scripts/04_make_sim_data.py
	$(PYTHON) $(ROOT)/scripts/04_make_sim_data.py $(DIMS)
	@touch $@

data: $(DATA_STAMP)

compile: $(TB_VVP)

$(TB_VVP): $(DATA_STAMP) $(RTL_SIM)/tb_ecg_conv_layer.v $(RTL_SIM)/axi_ram_model.v
	mkdir -p $(BUILD)
	$(IVERILOG) -g2005 -Wall -Wno-timescale -Idata -o $@ $(SRCS)

run: $(TB_VVP)
	$(VVP) $(TB_VVP) | tee $(BUILD)/run.log

check: run
	@grep -q "RESULT: PASS" $(BUILD)/run.log

wave: $(TB_VVP)
	$(VVP) $(TB_VVP) +vcd | tee $(BUILD)/run.log

clean:
	rm -rf $(BUILD) data/.stamp output.hex tb_ecg_conv_layer.vcd
