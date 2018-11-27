#!/bin/bash

tooldir="$1"
docdir="$2"
scoredir="$3"

function scores {
	infn="$docdir/$1.tsv"
	outfn="$@"
	outfn="$scoredir/${outfn// /}.tsv"
	perl -I"$tooldir/modules" "$tooldir/tap.pl" "${@:2}" -i "$infn" | tail -n +5 | head -n -6 > $outfn
}

# "normal" lists
for opts in {"-k "{5,12}" "{,"-q .33"},"-t "{1,.5}}; do
	for name in {short,long,single,asc,weighted,perfectorder,no_T_q}; do
		scores $name $opts
	done
done

# list with weights, unweighted
scores weighted -k 5 -u

# list with ascending E-values, explicitly flagged
scores asc -t 1 -m asc

# list without records
scores empty -t 1 -m asc
scores empty -t 1 -m desc

# list without errors
scores oracle -t 1
scores oracle -t .5

# pad-insufficient flag
scores oracle -k 5 -p
scores short -k 50 -p
