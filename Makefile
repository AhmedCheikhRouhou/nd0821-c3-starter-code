install:
	pip install --upgrade pip &&\
		pip install -r starter/requirements.txt

test:
	pytest -vv

format:
	black starter/*.py starter/starter/ml/*.py

lint:
	flake8 --ignore=E303,E302,E226  --max-line-length=200 starter/*.py starter/starter/ml/*.py

dvc:
	dvc pull -r storage

all: install lint test