# TAP-k -- Python module for Threshold Average Precision

Threshold Average Precision is a metric for evaluating IR applications that present retrieval lists with a ranking score to the end user.
It is described in the following publication:

> Hyrum D. Carroll, Maricel G. Kann, Sergey L. Sheetlin, and John L. Spouge: "Threshold Average Precision (TAP-_k_): A Measure of Retrieval Efficacy Designed for Bioinformatics" (2010). Bioinformatics 26(14):1708-1713. DOI: 10.1093/bioinformatics/btq270

This Python module is based on the authors' reference implementation written in Perl, which can be found at http://www.ncbi.nlm.nih.gov/CBBresearch/Spouge/html.ncbi/tap/.


## Installation

Use `pip` for a system-wide installation:

    pip3 install git+https://github.com/OntoGene/TAP-k.git

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

The input file format is the same as for the original program, which is described [here](https://www.ncbi.nlm.nih.gov/CBBresearch/Spouge/html_ncbi/html/tap/help.html).
All output is written to STDOUT.
The output format can be changed by specifying format strings (options `-f` and `-r`).

The module can also be used as a library:

```python
>>> import tapk
>>> tapk.run(["retrieval-list.tsv"], k=5)
EPQ (threshold at 0.5 quantile)	unweighted mean TAP
5 (0.6545522081334377)	0.5664
```
