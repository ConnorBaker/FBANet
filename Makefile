all: format lint

format:
	ruff format --preview

lint:
	ruff check --preview --fix

env:
	micromamba create -f env.yaml -y
