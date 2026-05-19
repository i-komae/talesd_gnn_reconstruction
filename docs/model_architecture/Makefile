DOC_DIR := $(abspath .)
BUILD_DIR := $(DOC_DIR)/build
TEX := main.tex
PDF := $(BUILD_DIR)/model_architecture.pdf
TEX_ENV := TEXMFVAR=$(BUILD_DIR)/texmf-var TEXMFCONFIG=$(BUILD_DIR)/texmf-config

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) $(wildcard fig/*.tikz) $(wildcard fig/*.pdf) $(wildcard fig/results/*.pdf) | $(BUILD_DIR)
	$(TEX_ENV) lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
		-output-directory=$(BUILD_DIR) -jobname=model_architecture $(TEX)
	$(TEX_ENV) lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
		-output-directory=$(BUILD_DIR) -jobname=model_architecture $(TEX)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(BUILD_DIR)/texmf-var $(BUILD_DIR)/texmf-config

clean:
	rm -rf $(BUILD_DIR)
