build:
	cd trex/models/liblinear; make clean; make all; cd -

get_deps:
	pip3 install -r requirements.txt

all: get_deps build