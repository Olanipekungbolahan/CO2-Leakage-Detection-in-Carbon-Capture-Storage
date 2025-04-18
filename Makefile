.PHONY: setup test train serve format lint clean docker-up docker-down

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/ --cov=src --cov-report=xml --cov-report=html

train:
	python src/train.py

serve:
	python src/serve.py

format:
	black src/ tests/
	isort src/ tests/

lint:
	pylint src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down -v