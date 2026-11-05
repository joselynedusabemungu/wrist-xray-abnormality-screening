PYTHON ?= python3
MURA_ROOT ?= $(CURDIR)/MURA-v1.1
export MURA_ROOT

.PHONY: help install install-onnx data-check train-classical train-resnet train-all smoke \
        compare operating-point calibration export-onnx notebook fill-docs check-tokens app all clean

help:
	echo "install           install pinned dependencies"
	echo "data-check        verify dataset, print real counts, assert no patient overlap"
	echo "train-classical   grayscale+LR and HOG+LR baselines"
	echo "train-resnet      ResNet18 transfer learning (CPU works, but slowly)"
	echo "train-all         both of the above"
	echo "smoke             tiny end-to-end run (<=64 train images, 1 epoch) to test the plumbing"
	echo "compare           outputs/model_comparison.{md,csv}"
	echo "operating-point   recall-target threshold + PR curve (all models)"
	echo "calibration       reliability diagram + abstention band (all models)"
	echo "export-onnx       optional ONNX export + parity check"
	echo "notebook          execute the notebook top-to-bottom, then scan for leftover tokens"
	echo "fill-docs         fill README/docs placeholders from outputs/ (never invents numbers)"
	echo "check-tokens      fail if README/docs still contain unfilled placeholders"
	echo "app               launch the Streamlit demo"
	echo "all               scripts/run_all.sh"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"

install-onnx:
	$(PYTHON) -m pip install -r requirements-onnx.txt

data-check:
	$(PYTHON) -m src.mura_utils

train-classical:
	$(PYTHON) -m src.train_classical

train-resnet:
	$(PYTHON) -m src.train_resnet

train-all: train-classical train-resnet

smoke:
	$(PYTHON) -m src.train_classical --max-train-images 64
	$(PYTHON) -m src.train_resnet --max-train-images 64 --epochs 1
	$(PYTHON) -m src.compare_models

compare:
	$(PYTHON) -m src.compare_models

operating-point:
	$(PYTHON) -m src.operating_point --model resnet
	$(PYTHON) -m src.operating_point --model hog
	$(PYTHON) -m src.operating_point --model grayscale

calibration:
	$(PYTHON) -m src.calibration --model resnet
	$(PYTHON) -m src.calibration --model hog
	$(PYTHON) -m src.calibration --model grayscale

export-onnx:
	$(PYTHON) -m src.export_onnx

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/MURA_Wrist_Classification.ipynb
	$(PYTHON) scripts/check_notebook_tokens.py notebooks/MURA_Wrist_Classification.ipynb

fill-docs:
	$(PYTHON) -m src.fill_placeholders

check-tokens:
	$(PYTHON) -m src.fill_placeholders --check

app:
	$(PYTHON) -m streamlit run app.py

all:
	bash scripts/run_all.sh

clean:
	rm -rf outputs/artifacts outputs/*.json outputs/*.png outputs/*.md outputs/*.csv outputs/*.npz
