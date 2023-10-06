all: build
.PHONY: build install

build:
	@python3 setup.py bdist_wheel

install: build
	@pip3 install dist/infscale*.whl

uninstall:
	@pip3 uninstall -y infscale

clean:
	@rm -rf build dist infscale.egg-info
