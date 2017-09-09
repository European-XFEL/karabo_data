# makefile used for testing

install:
	python3 -m pip install .

test:
	python3 -m pytest -v tests

build-docker-image:
	docker build -t image .

ci-docker:
	# uses normal targets, but executes in container
	docker run -v `pwd`:/io image make install test

	# make the install part of the testing process ^
