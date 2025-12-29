clean:
	docker rm -f $$(docker ps -qa)

clean-images:
	docker rmi $$(docker images -f "dangling=true" -q)

build:
	docker build -t desom_pytorch .

# make run accelerator=cuda devices=1
run:
	docker run -it --gpus all --rm -v .:/app --shm-size=24g \
	-p 6006:6006 \
	-e ACCELERATOR=$(accelerator) \
	-e DEVICES=$(devices) \
	desom_pytorch /bin/bash

train:
	PYTHONPATH=./:$PYTHONPATH python experiments/benchmarking/train_$(model).py --config configs/$(model)/$(model)_$(dataset).yaml

test:
	PYTHONPATH=./:$PYTHONPATH python experiments/tests/test_$(model).py --config configs/$(model)/$(model)_$(dataset).yaml

unit-test:
	PYTHONPATH=./:$PYTHONPATH python experiments/tests/unit_test.py run
