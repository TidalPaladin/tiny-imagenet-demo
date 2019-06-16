.PHONY: build build-example doc test clean dataset build-dataset

IMG_NAME='tiny-imagenet-demo'
LIB_NAME='tin'

DATA_SRC="/mnt/valak/documents/imagenet/ILSVRC/Data/CLS-LOC/train"
DATA_DEST="/home/tidal/Documents/tiny-imagenet-demo/dataset"
DATA_IMG='tiny-imagenet-dataset'

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

test:
	docker run -it \
		-v ${PWD}/test:/test \
		-v ${PWD}/${LIB_NAME}:/${LIB_NAME} \
		${IMG_NAME}:test \
		--cov=/${LIB_NAME} \
		${pytest_args} /test

build-dataset:
	docker build --tag=${DATA_IMG} scripts/

dataset:
	docker run -it \
		-v /home/tidal/Documents/tiny-imagenet-demo/scripts:/app \
		-v ${DATA_SRC}:/app/src \
		-v ${DATA_DEST}:/app/dest \
		${DATA_IMG} \
