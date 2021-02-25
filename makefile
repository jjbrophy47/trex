clean:
	rm -rf .catboost_info

get_deps:
	pip3 install -r requirements.txt

build:
	cd trex/models/; rm -rf *.so *.c *.html build/ __pycache__; python3 setup.py build_ext --inplace; cd -

all: clean get_deps build
