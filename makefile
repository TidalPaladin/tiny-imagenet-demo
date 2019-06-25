.PHONY: build build-example doc test clean

IMG_NAME='tiny-imagenet-demo'
LIB_NAME='tin'

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -wholename '*/.pytest_cache' -exec rm -rf {} +

build:
	docker build --tag=${IMG_NAME} --target=base .
	docker build \
		--tag=${IMG_NAME}:test \
		--build-arg cache=${shell date +%Y-%m-%d:%H%M:%s}  \
		.

train:
	docker run -it \
		-v ${PWD}:/app \
		${IMG_NAME}

test:
	docker run -it \
		-v ${PWD}/test:/test \
		-v ${PWD}/${LIB_NAME}:/${LIB_NAME} \
		${IMG_NAME}:test \
		--cov=/${LIB_NAME} \
		${pytest_args} /test
