#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2016


'''
Compute Threshold Average Precision at (a median of) k errors per query.
'''


__version__ = '1.0b'


import re
import sys
import math
import numbers
import argparse
from collections import namedtuple


# Constants.
ASC = 'asc'
DESC = 'desc'

# Defaults.
QUANTILE = 0.5  # median
SUMMARY_FMT_E0 = 'Threshold\t{u} mean TAP\n{e0}\t{tap:.4f}'
SUMMARY_FMT_K = ('EPQ (threshold at {q} quantile)\t{u} mean TAP\n'
                 '{k} ({e0})\t{tap:.4f}')
QUERY_FMT = '{query} ({weight:g})\t{tap:.4f}'
QUERY_FMT_UNW = '{query}\t{tap:.4f}'


def main():
    '''
    Run from commandline.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    inparams = ap.add_argument_group('input')
    inparams.add_argument(
        '-i', '--retrieval-lists',
        dest='infiles', nargs='+', default=['-'], metavar='PATH',
        help='one or more input files (UTF-8) containing retrieval lists '
             '(default: STDIN)')
    procparams = ap.add_argument_group('parameters')
    eparam = procparams.add_mutually_exclusive_group(required=True)
    eparam.add_argument(
        '-k', type=posint, metavar='N', nargs='+',
        help='number of errors per query for calculating E0')
    eparam.add_argument(
        '-t', dest='e0', type=float, metavar='F', nargs='+',
        help='threshold score or E value (bypass `k`)')
    procparams.add_argument(
        '-m', '--monotonicity', choices=(ASC, DESC),
        help='descending scores or ascending E values? '
             '(default: the lists determine)')
    procparams.add_argument(
        '-q', '--quantile', type=zerotoone, default=QUANTILE, metavar='F',
        help='quantile q, 0.0 < q <= 1.0 '
             '(default: the median, 0.5)')
    procparams.add_argument(
        '-u', '--unweighted', action='store_true',
        help='ignore all weights, to perform an unweighted calculation')
    procparams.add_argument(
        '-p', '--pad-insufficient', action='store_true',
        help='pad insufficient retrieval lists with irrelevant records')
    outparams = ap.add_argument_group('output formatting')
    outparams.add_argument(
        '-s', '--summary-only', action='store_false',
        dest='show_query_wise_result',
        help='suppress query-wise results')
    outparams.add_argument(
        '-f', '--summary-format', metavar='FMT', type=unescape_backslashes,
        help='format string for the result summary. '
             'Available fields: k, e0, tap, q[uantile], u[nweighted]')
    outparams.add_argument(
        '-Q', '--query-format', metavar='FMT', type=unescape_backslashes,
        help='format string for a separate result line per query. '
             'Available fields: query, tap, weight, T_q '
             '(default: {!r} for weighted, {!r} for unweighted TAP)'
             ''.format(QUERY_FMT, QUERY_FMT_UNW))
    ap.add_argument(
        '-V', '--version', action='version',
        version='TAP-k {}'.format(__version__))

    args = ap.parse_args()
    try:
        run(output=sys.stdout, **vars(args))
    except InputError as e:
        ap.exit(str(e))  # no usage message
    except BrokenPipeError:
        # Die silently when the output is truncated by Unix head or less.
        pass


def run(infiles, k=None, e0=None, unweighted=False, monotonicity=None,
        **params):
    '''
    Run with keyword args.
    '''
    # Parse and sanity-check the retrieval lists.
    retlists = [rl for src in infiles for rl in parserecords(src, unweighted)]
    monotonicity = sanity_check(retlists, monotonicity)
    params.update(retlists=retlists, ascending=monotonicity == ASC)

    # Iterate over multiple values of E0 or k, depending on what is given.
    if e0 is not None:
        elems = e0
        p = 'e0'
    elif k is not None:
        elems = k
        p = 'k'
    else:
        raise ValueError('either k or e0 must be specified')
    if isinstance(elems, numbers.Number):
        elems = (elems,)

    for e in elems:
        params[p] = e
        run_one(**params)


def run_one(quantile=QUANTILE, output=sys.stdout, show_query_wise_result=False,
            summary_format=None, query_format=None, **params):
    '''
    Evaluate and output TAP for one value of k or E0.
    '''
    # Result computation.
    result = evaluate(quantile=quantile, **params)

    # Output formatting.
    unweighted = all(r.weight == 1.0 for r in result.queries)
    params.update(
        tap=result.tap, k=result.k, e0=result.e0, q=quantile,
        u='unweighted' if unweighted else 'weighted')
    # Summary.
    if summary_format is None:
        summary_format = SUMMARY_FMT_E0 if result.k is None else SUMMARY_FMT_K
    print(summary_format.format_map(params), file=output)
    # Query-wise results.
    if show_query_wise_result:
        if query_format is None:
            query_format = QUERY_FMT_UNW if unweighted else QUERY_FMT
        print('\nIndividual results for each query:', file=output)
        for query in result.queries:
            print(query_format.format_map(query._asdict()), file=output)
        print(file=output)


def evaluate(retlists, quantile=QUANTILE, e0=None, pad_insufficient=False,
             **params):
    '''
    Calculate TAP-k for multiple queries.
    '''
    if e0 is None:
        e0 = determine_E0(retlists, quantile=quantile,
                          pad_insufficient=pad_insufficient, **params)
    return evaluate_e0(retlists, e0, **params)


def evaluate_e0(retlists, e0, ascending, k=None):
    '''
    Calculate TAP for a given threshold, for multiple queries.
    '''
    if ascending:
        def past_E0(score):
            'E value is beyond E0.'
            return score > e0
    else:
        def past_E0(score):
            'Threshold score is beyond E0.'
            return score < e0
    e0_nan = math.isnan(e0)
    results = [QueryResult(r.query, tap(r, past_E0, e0_nan), r.weight, r.T_q)
               for r in retlists]
    avg_tap = (sum(r.tap * r.weight for r in results) /
               sum(r.weight for r in results))
    return Result(avg_tap, k, e0, results)

Result = namedtuple('Result', 'tap k e0 queries')
QueryResult = namedtuple('QueryResult', 'query tap weight T_q')


def tap(records, past_E0, e0_nan):
    '''
    Threshold average precision for one query.
    '''
    summed_precision = 0.0
    rel_count = 0  # number of relevant records seen so far
    retrieved = 0  # number of records considered
    for relevance, score in records:
        if past_E0(score):
            # Exit the loop without updating the retrieved count.
            break
        retrieved += 1
        # Sum precision value at each relevant record.
        if relevance:
            rel_count += 1
            summed_precision += rel_count / retrieved

    if not records.T_q:
        # No ground-truth targets: Retrieving nothing is the right thing to do.
        summed_precision = 1.0 / (retrieved + 1)
    elif retrieved and not e0_nan:
        # The sentinel needs to be added to the average
        # (a second time if the last retrieved record was relevant), unless
        #  - there were no records at all, in which case the summed
        #    precision is 0 by definition, or
        #  - E0 is NaN as a result of truncated retrieval lists in
        #    combination with the --pad-insufficient option.
        summed_precision += rel_count / retrieved

    # Divide by the total number of relevant records,
    # plus one for the sentinel.
    return summed_precision / (records.T_q + 1)


def determine_E0(retlists, k, quantile, ascending, pad_insufficient):
    '''
    Determine E_k(A) based on the retrieved records.

    Args:
        retlists (iterable of iterable of (bool, float)):
            the records for each query, consisting of
            pairs <relevance label, E value>
        k (int): number of errors per query at quantile
        quantile (float): between 0 and 1; eg. 0.5 for
            median
        ascending (bool): True if the E values increase
            from best to worst.
    '''
    # Collect the E value at k errors for each query.
    E_k = []
    total_weights = 0.0
    for records in retlists:
        total_weights += records.weight
        errors = 0
        for relevance, score in records:
            if not relevance:
                errors += 1
                if errors >= k:
                    E_k.append((score, records.weight))
                    break
    # If the inner loop ends without a break, then there are
    # less than k errors in this query.
    # In this case E0 could be arbitrarily generous without introducing
    # more than k errors, which is equivalent to placing it at the end
    # of the sorted list.
    # However, the quantile calculation walks from the start, so we can
    # simply skip those.

    # Sort the scores from best to worst.
    E_k.sort(reverse=not ascending)
    try:
        E0 = weighted_quantile(E_k, quantile, total_weights)
    except InputError:
        # E0 can't be determined.
        if pad_insufficient:
            # Compute TAP of the complete retrieval lists (no cutoff at E0).
            E0 = math.nan
        else:
            # Re-raise with a proper message (the callee didn't have all info).
            raise InputError(
                'Fewer than {} of the retrieval lists have {} errors.'
                .format(quantile, k))
    return E0


def weighted_quantile(weighted_scores, quantile, total_weights):
    '''
    Walk through the sorted scores until quantile is reached.
    '''
    quantile *= total_weights  # spread q to the absolute weight scale
    quantile *= 1.0 - 1e-12  # don't miss it because of imprecision
    summed_weight = 0.0
    for s, w in weighted_scores:
        summed_weight += w
        if summed_weight >= quantile:
            return s
    # If we get here, then we fell off the end of the list --
    # there are not enough retrieval lists with k errors.
    raise InputError  # construct the message in the caller


def sanity_check(retlists, monotonicity):
    '''
    Thoroughly check the retrieval lists and determine monotonicity.

    Any failure raises an InputError.
    On success, return the monotonicity (ASC or DESC).

    Check the following:
    - at least 1 retrieval list given
    - number of relevant records is less than or euqal to T_q
    - monotonicity
      * is determined (either through -m or the data)
      * is given in each list
      * is consistent among all lists
      * if given through -m, is consistent with the data
    '''
    if not retlists:
        raise InputError('no retrieval list found')
    for records in retlists:
        records.check_rel_count()
        monotonicity = records.check_monotonicity(monotonicity)
    if monotonicity is None:
        raise InputError(
            'Every retrieval list was empty or repeated the same score or '
            'E-value,\nin which case you must specify through the -m/'
            '--monotonicity option\n'
            'whether the lists are ascending or descending.')
    return monotonicity


def parserecords(source, unweighted=False):
    '''
    Iterate over RetrievalList instances parsed from plain-text.
    '''
    current = None
    stream = enumerate(smartopen(source))
    for i, line in stream:
        try:
            current.add(line, i)
        except AttributeError:
            # current is None: Start a new retrieval list.
            if line.strip():
                # Consume the next line as well (without its line number).
                current = RetrievalList.incremental_factory(
                    line, next(stream)[1], source, i, unweighted)
        except BlankLineSignal:
            # This retrieval list has ended.
            # Yield it and reset current (so it won't be yielded again,
            # since multiple blank lines are allowed).
            yield current
            current = None
    if current is not None:
        # The last list might still need to be yielded.
        yield current


def smartopen(source):
    '''
    Open file if necessary and iterate over its lines.
    '''
    if source == '-':
        # STDIN.
        yield from sys.stdin
    elif isinstance(source, str):
        # File name.
        with open(source, encoding='utf8') as f:
            yield from f
    else:
        # Anything else: try to iterate over.
        yield from source


class RetrievalList:
    '''
    A list of rated records and some metadata.
    '''
    __slots__ = ('source', 'query', 'weight', 'T_q', 'records')

    def __init__(self, source, query, T_q, records, weight=1.0):
        self.source = source
        self.query = query
        self.weight = weight
        self.T_q = T_q
        self.records = records

    @classmethod
    def incremental_factory(cls, line1, line2, src, no, unweighted=False):
        '''
        Constructor for incremental building through parsing.
        '''
        # Parse the header lines and be specific about any failure.
        try:
            query, weight = line1.split()
        except ValueError as e:
            if str(e).startswith('too many values'):
                raise InputFormatError(
                    'The line for a unique identifier '
                    'should have at most 2 columns.', src, no, line1)
            # If the weight was missing, it defaults to 1.
            query = line1.strip()
            weight = 1.0
        else:
            try:
                weight = float(weight)
                if weight <= 0:
                    raise ValueError()
            except ValueError:
                raise InputFormatError(
                    'Column 2 in the line for a unique identifier '
                    'should be a positive weight.', src, no, line1)
        try:
            T_q = int(line2)
            if T_q < 0:
                raise ValueError()
        except ValueError:
            raise InputFormatError(
                'The line containing the number of relevant records '
                'should be a non-negative integer', src, no+1, line2)
        # Construct a RetrievalList with a (yet) empty records list.
        if unweighted:
            weight = 1.0
        return cls(src, query, T_q, [], weight)

    def add(self, line, no):
        'Parse and add an input record.'
        try:
            relevance, score, *_ = line.split()
            relevance = int(relevance)
            if relevance not in (0, 1):
                raise ValueError()
            score = float(score)
        except ValueError:
            if not line.strip():
                raise BlankLineSignal()
            else:
                raise InputFormatError(
                    'Column 1 should have shown record relevancy as 0 or 1.\n'
                    'Column 2 should have shown the record score as a float.',
                    self.source, no, line)
        self.records.append((bool(relevance), score))

    def check_rel_count(self):
        '''
        The number of records marked as relevant must not exceed T_q.
        '''
        total = sum(rel for rel, _ in self)
        if total > self.T_q:
            raise InputValueError(
                'too many records marked as relevant (found {}, T_q = {})'
                .format(total, self.T_q),
                self.source, self.query)

    def check_monotonicity(self, given):
        '''
        Check and (if necessary) determine monotonicity.
        '''
        found = None
        for a, b in self._bigram_scores():
            found = self._monotonicity(a, b)
            if found is None:
                continue
            elif given is None:
                given = found
            elif found != given:
                raise InputValueError(
                    'monotonicity violation (given {}, found {})'
                    .format(given, found),
                    self.source, self.query)
        return found or given

    def _bigram_scores(self):
        if len(self) < 2:
            # Short-circuit for insufficient lists.
            return
        i = (score for _, score in self)
        first = next(i)
        for second in i:
            yield first, second
            first = second

    @staticmethod
    def _monotonicity(a, b):
        if a < b:
            return ASC
        elif a > b:
            return DESC
        else:
            return None

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


class BlankLineSignal(Exception):
    'Empty input line.'

class InputError(Exception):
    'Unspecific input-related problem.'

class _LocatableInputError(InputError):
    'Base class for errors with a specific source.'
    def __init__(self, message, source, *args):
        if not isinstance(source, str):
            # Limit source description to 99 characters.
            source = repr(source)
            if len(source) >= 100:
                source = '{}...{}'.format(source[:48], source[-48:])
        super().__init__(message, source, *args)
        self.source = source
        self.message = message

class InputFormatError(_LocatableInputError):
    'Input text could not be parsed.'
    def __init__(self, message, source, line_number, line):
        super().__init__(message, source, line_number, line)
        self.line_number = line_number
        self.line = line

    def __str__(self):
        location = ('Offending input file: {}, line: {}'
                    .format(self.source, self.line_number))
        return '\n'.join((self.message, location, self.line))

class InputValueError(_LocatableInputError):
    'Input is incomplete or inconsistent.'
    def __init__(self, message, source, query):
        super().__init__(message, source, query)
        self.query = query

    def __str__(self):
        return ('File {}, query {}: {}'
                .format(self.source, self.query, self.message))


def posint(expr):
    '''
    Make sure expr is a positive integer.
    '''
    k = int(expr)
    if k <= 0:
        raise argparse.ArgumentTypeError(
            '{} is not a positive integer'.format(expr))
    return k


def zerotoone(expr):
    '''
    Make sure expr is a float in the interval ]0..1].
    '''
    q = float(expr)
    if not 0 < q <= 1:
        raise argparse.ArgumentTypeError(
            'violation of 0 < q <= 1 (given: {})'.format(expr))
    return q


def unescape_backslashes(expr):
    '''
    Process a few backslash sequences.

    Replace \t, \n, and \r in format strings
    with the actual tab/newline characters,
    unless preceded by another backslash.
    '''
    mapping = {r'\t': '\t', r'\n': '\n', r'\r': '\r'}
    def map_(match):
        'Map the sequence to its character.'
        return mapping[match.group()]
    expr = re.sub(r'''(?<!\\)  # negative lookbehind: no preceding backslash
                      \\[tnr]  # escaped tab, LF, or CR''',
                  map_, expr, flags=re.VERBOSE)
    # Unescape backslashes.
    expr = expr.replace('\\\\', '\\')
    return expr


if __name__ == '__main__':
    main()
