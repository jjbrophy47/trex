build:
	cd trex/models/liblinear; make clean; make all; cd -

get_deps:
	pip3 install -r requirements.txt

clean:
	rm -rf .trex .catboos_info

all: get_deps build