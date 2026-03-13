.PHONY: test exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp10 exp11 exp12 exp-fingerprint clean

test:
	pytest tests/ -v

exp01:
	python experiments/exp01_validate.py

exp02:
	python experiments/exp02_example_b.py

exp03:
	python experiments/exp03_straightness.py

exp04:
	python experiments/exp04_enumeration.py

exp05:
	python experiments/exp05_ordering.py

exp06:
	python experiments/exp06_temporal_profiles.py

exp07:
	python experiments/exp07_dpo_null_model.py

exp10:
	python experiments/exp10_perturbation_response.py

exp11:
	python experiments/exp11_perturbation_ratio.py

exp12:
	python experiments/exp12_dehn_twist.py

exp-fingerprint:
	python experiments/exp_physics_fingerprint.py

reproduce-paper-1: exp01 exp03
	@echo "Paper 1 results reproduced."

reproduce-paper-2: exp01 exp02
	@echo "Paper 2 results reproduced."

reproduce-paper-3:
	@echo "Paper 3 is purely theoretical — no experiments to reproduce."

reproduce-paper-4: exp04 exp06 exp10 exp11 exp12
	@echo "Paper 4 results reproduced."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
