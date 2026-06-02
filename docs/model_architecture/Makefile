DOC_DIR := $(abspath .)
BUILD_DIR := $(DOC_DIR)/build
JP_TEX := main.tex
EN_TEX := main_en.tex
JP_PDF := $(BUILD_DIR)/model_architecture.pdf
EN_PDF := $(BUILD_DIR)/model_architecture_en.pdf
TEX_ENV := TEXMFVAR=$(BUILD_DIR)/texmf-var TEXMFCONFIG=$(BUILD_DIR)/texmf-config

.PHONY: all ja en clean

all: ja en

ja: $(JP_PDF)

en: $(EN_PDF)

$(JP_PDF): $(JP_TEX) $(wildcard fig/*.tikz) $(wildcard fig/*.pdf) $(wildcard fig/results/*.pdf) $(wildcard fig/input_distributions/*.pdf) | $(BUILD_DIR)
	$(TEX_ENV) lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
		-output-directory=$(BUILD_DIR) -jobname=model_architecture $(JP_TEX)
	$(TEX_ENV) lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
		-output-directory=$(BUILD_DIR) -jobname=model_architecture $(JP_TEX)

$(EN_PDF): $(EN_TEX) $(wildcard fig/*.tikz) $(wildcard fig/*.pdf) $(wildcard fig/results/*.pdf) $(wildcard fig/input_distributions/*.pdf) | $(BUILD_DIR)
	$(TEX_ENV) lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
		-output-directory=$(BUILD_DIR) -jobname=model_architecture_en $(EN_TEX)
	$(TEX_ENV) lualatex -interaction=nonstopmode -halt-on-error -file-line-error \
		-output-directory=$(BUILD_DIR) -jobname=model_architecture_en $(EN_TEX)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(BUILD_DIR)/texmf-var $(BUILD_DIR)/texmf-config

clean:
	rm -rf $(BUILD_DIR)
