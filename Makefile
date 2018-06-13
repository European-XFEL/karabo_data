# makefile used for testing

install:
	python3 -m pip install .[test]

dev-install:
	python3 -m pip install -U -e .[test]

test:
	python3 -m pytest -v --nbval-lax --cov=karabo_data

docker-build:
	@# build docker image
	docker build -t image .

docker-test:
	@# uses normal targets, but executes in container

	@# use matplotlib backend that does not require X (or TK)
	echo "backend      : Agg" > matplotlibrc

	@# make the install part of the testing process ^
	docker run -tv `pwd`:/io image make install test

	@# remove matplotlibrc file again (or should we leave it?)
	@# would be better to do this via environment variable in
	@# container
	rm -v matplotlibrc
