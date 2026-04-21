.PHONY: serve generate client concurrent eval perf perf-long validate improve install

serve:
	python serve/serve.py

generate:
	python serve/generate.py

client:
	python serve/client.py

concurrent:
	python serve/client.py --concurrent 5

eval:
	cd eval_runner && python run_eval.py --limit 100

perf:
	python perf/load_test.py --n 20 --concurrency 5

perf-long:
	python perf/load_test.py --n 20 --concurrency 5 --long

validate:
	python guardrails/validate.py

improve:
	cd improve && bash eval.sh

install:
	pip install -r requirements.txt
