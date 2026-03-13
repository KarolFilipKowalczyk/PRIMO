.PHONY: test exp01 exp02 exp03 exp04 exp05 exp06 clean

test:
	pytest tests/ -v

exp01:
	python experiments/exp01_validate.py

exp02:
	python experiments/exp02_example_b.py

exp03:
	python experiments/exp03_straightness.py

exp04:
	python experiments/exp04_enumerate.py

exp05:
	python experiments/exp05_ordering.py

exp06:
	python experiments/exp06_temporal.py

reproduce-paper-1: exp01 exp03
	@echo "Paper 1 results reproduced."

reproduce-paper-2: exp01 exp02
	@echo "Paper 2 results reproduced."

reproduce-paper-3:
	@echo "Paper 3 is purely theoretical — no experiments to reproduce."

reproduce-paper-4: exp04 exp05 exp06
	@echo "Paper 4 results reproduced."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
