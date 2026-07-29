.PHONY: help install dev help-cli toy smoke test check clean

help:
	@echo "Targets:"
	@echo "  install   Install runtime dependencies and package"
	@echo "  dev       Install runtime and development dependencies"
	@echo "  help-cli  Show help for all trainers"
	@echo "  toy       Generate synthetic example data"
	@echo "  smoke     Run minimal training examples"
	@echo "  simulate  Run the minimal simulation example"
	@echo "  test      Compile and run tests"
	@echo "  check     Check release hygiene"
	@echo "  clean     Remove generated data and outputs"

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

dev:
	python -m pip install -r requirements.txt
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

help-cli:
	python train_rf.py --help
	python train_crossnn.py --help
	python train_mpcnet.py --help
	python scripts/simulation/generate_cross_platform_in_silico_beta.py --help

toy:
	python examples/make_toy_data.py

smoke:
	bash examples/run_all_smoke.sh

simulate:
	python examples/make_simulation_reference.py
	bash examples/run_simulation.sh

test:
	python -m compileall -q mbmmc tools scripts examples tests
	pytest -q

check:
	python -m tools.check_release

clean:
	rm -rf examples/data outputs .pytest_cache
