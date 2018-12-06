# TAP-k -- Python module for Threshold Average Precision

Threshold Average Precision is a metric for evaluating IR applications that present retrieval lists with a ranking score to the end user.
It is described in the following publication:

> Hyrum D. Carroll, Maricel G. Kann, Sergey L. Sheetlin, and John L. Spouge: "Threshold Average Precision (TAP-_k_): A Measure of Retrieval Efficacy Designed for Bioinformatics" (2010). Bioinformatics 26(14):1708-1713. DOI: 10.1093/bioinformatics/btq270

This Python module is based on the authors' reference implementation written in Perl, which can be found at http://www.ncbi.nlm.nih.gov/CBBresearch/Spouge/html.ncbi/tap/.


## Installation

Use `pip` for a system-wide installation:

    pip3 install TAP-k

Or simply download the stand-alone script [tapk.py](https://github.com/OntoGene/TAP-k/blob/master/tapk.py).


## Requirements

`TAP-k` runs on Python 3 (Python 3.4 or higher recommended).  
No other requirements.


## Usage

Installing via `pip` creates an executable script *TAP-k* to run from the command line:

    TAP-k [options]

If you just downloaded the stand-alone module, use the following call:

    python3 tapk.py [options]

Run `TAP-k --help` to see a short description of all available options.

Example call with minimal output:

    $ TAP-k -i test/short.tsv -k 5 -s
    EPQ (threshold at 0.5 quantile)	unweighted mean TAP
    5 (0.6545522081334377)	0.5664

The input file format is the same as for the original program, which is described [here](https://www.ncbi.nlm.nih.gov/CBBresearch/Spouge/html_ncbi/html/tap/help.html).
All output is written to STDOUT.
The output format can be changed by specifying format strings (options `-f` and `-Q`).

The module can also be used as a library:

```pycon
>>> import tapk
>>> retlists = ['test/retlists/{}.tsv'.format(fn)
                for fn in ('weighted', 'single')]
>>> result = tapk.tapk(retlists, k=5)
>>> result.tap
0.1311308349769888
>>> result.e0
0.7418847867396157
>>> result.queries
[QueryResult(query='23817572', tap=0.22749287749287747, weight=3.0, T_q=8),
 QueryResult(query='12954810', tap=0.0, weight=1.0, T_q=1),
 QueryResult(query='20729916', tap=0.08055555555555555, weight=2.0, T_q=1),
 QueryResult(query='21519793', tap=0.0, weight=5.0, T_q=1),
 QueryResult(query='7787496', tap=0.5833333333333334, weight=1.0, T_q=1),
 QueryResult(query='1303262', tap=0.27777777777777773, weight=1.0, T_q=2)]
```
