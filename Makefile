build:
	python3 setup.py build

install:
	python3 setup.py install

clean:
	rm -r build dist TAP_k.egg_info 2> /dev/null || exit 0

clean-test:
	rm -r test/ref-scores test/ref-tool 2> /dev/null || exit 0

.PHONY: test
test: test/ref-scores
	python3 test/test.py -v

test/ref-scores: test/ref-tool/tap.pl
	mkdir -p $@
	test/create-ref-scores.sh $(<D) test/retlists $@

test/ref-tool/tap.pl: test/TAP_1.8.zip
	unzip -d test $< "TAP_1.8/*" || exit 0
	mv test/TAP_1.8 $(@D)

.INTERMEDIATE: test/TAP_1.8.zip
test/TAP_1.8.zip:
	wget -O $@ ftp://ftp.ncbi.nih.gov/pub/spouge/web/software/TAP_1.8/TAP_1.8.zip
