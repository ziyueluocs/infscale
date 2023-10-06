all: build
.PHONY: install

clean:
	@rm -rf build dist infscale.egg-info

install:
	@pip3 install .

uninstall:
	@pip3 uninstall -y infscale

reinstall: clean uninstall install
