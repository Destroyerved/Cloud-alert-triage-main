.PHONY: run test docker-build docker-run validate smoke infer

run:
	python -m uvicorn server.app:app --reload --port 7860

test:
	pytest tests/ -v

docker-build:
	docker build -t cloud-alert-triage .

docker-run:
	docker run -p 7860:7860 cloud-alert-triage

validate:
	python -m openenv.cli validate

smoke:
	python scripts/smoke_test.py

infer:
	python inference.py
