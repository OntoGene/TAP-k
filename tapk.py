#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2016


'''
Compute Threshold Average Precision at (a median of) k errors per query.
'''


import re
import sys
import numbers
import argparse
from collections import namedtuple


def main():
    '''
    Run from commandline.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '-i', '--retrieval-lists', default='-', metavar='PATH',
        type=argparse.FileType('r'), dest='infile',
        help='a single input file containing retrieval lists '
             '(default: STDIN)')
    eparam = ap.add_mutually_exclusive_group(required=True)
    eparam.add_argument(
        '-k', type=int, metavar='N', nargs='+',
        help='number of errors per query for calculating E0')
    eparam.add_argument(
        '-t', dest='e0', type=float, metavar='F', nargs='+',
        help='threshold score or E value (bypass `k`)')
    ap.add_argument(
        '-m', '--monotonicity', choices=('asc', 'desc'),
        help='descending scores or ascending E values? '
             '(default: the lists determine)')
    ap.add_argument(
        '-q', '--quantile', type=zerotoone, default=0.5, metavar='F',
        help='quantile q, 0.0 < q <= 1.0 '
             '(default: the median, 0.5)')
    ap.add_argument(
        '-u', '--unweighted', action='store_true',
        help='ignore all weights, to perform an unweighted calculation')
    ap.add_argument(
        '-f', '--result-format', metavar='FMT', type=unescape_backslashes,
        default='EPQ (threshold at {q} quantile)\t{u} mean TAP\n'
                '{k} ({e0})\t{tap:.4f}',
        help='format string for the overall result (default: %(default)r)')
    ap.add_argument(
        '-r', '--query-wise-result', metavar='FMT', type=unescape_backslashes,
        nargs='?', const='{query}\t{tap}',
        help='format string for a result line per query. '
             'Available fields: query, tap, weight, T_q '
             '(default: suppress query-wise results, '
             '-r without argument: %(const)r)')

    # TODO:
    # * -p pad insufficient retrieval list(s) with irrelevant records
    #       (E0 is NaN; no sentinel)
    # * more checks and error messages (eg. inconsistent monotonicity)
    # * if T_q == 0, an empty retrieval list should have a TAP of 1

    args = ap.parse_args()
    try:
        run(output=sys.stdout, **vars(args))
    except InputFormatError as e:
        ap.exit(str(e))  # no usage message
    except InputValueError as e:
        ap.error(str(e))
    except BrokenPipeError:
        # Die silently when the output is truncated by Unix head or less.
        pass


def run(infile, k, e0=None, unweighted=False, monotonicity=None, **params):
    '''
    Run with keyword args.
    '''
    retlists = list(parserecords(infile, unweighted))

    try:
        ascending = dict(asc=True, desc=False)[monotonicity]
    except KeyError:
        ascending = determine_monotonicity(retlists)

    params.update(retlists=retlists, ascending=ascending)

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


def run_one(output, result_format, query_wise_result, **params):
    '''
    Evaluate and output TAP for one value of k or E0.
    '''
    result = evaluate(**params)

    unweighted = all(r.weight == 1 for r in result.queries)
    params.update(
        tap=result.tap, k=result.k, e0=result.e0, q=params['quantile'],
        u='unweighted' if unweighted else 'weighted')
    print(result_format.format_map(params), file=output)
    if query_wise_result is not None:
        for query in result.queries:
            print(query_wise_result.format_map(query._asdict()), file=output)


def evaluate(retlists, quantile, e0=None, **params):
    '''
    Calculate TAP-k for multiple queries.
    '''
    if e0 is None:
        e0 = determine_E0(retlists, quantile=quantile, **params)
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
    results = [QueryResult(r.query, tap(r, past_E0), r.weight, r.T_q)
               for r in retlists]
    avg_tap = (sum(r.tap * r.weight for r in results) /
               sum(r.weight for r in results))
    return Result(avg_tap, k, e0, results)

Result = namedtuple('Result', 'tap k e0 queries')
QueryResult = namedtuple('QueryResult', 'query tap weight T_q')


def tap(records, past_E0):
    '''
    Threshold average precision for one query.
    '''
    # Sum precision value at each relevant record.
    summed_precision = 0
    rel_count = 0  # number of relevant records seen so far
    for i, (relevance, score) in enumerate(records, 1):
        if past_E0(score):
            # We are past the cutoff point E0.
            # The sentinel needs to be added to the average
            # (a second time if the last record was relevant).
            if rel_count:
                # No need to add 0 precision.
                # Also, avoid ZeroDivisionError in case of the first record.
                summed_precision += rel_count / (i-1)
            break
        if relevance:
            rel_count += 1
            summed_precision += rel_count / i
    else:
        # We reached the last record without passing E0.
        # This means we still need to add the sentinel.
        # (Unless there were no records at all, in which case the summed
        # precision is 0 by definition.)
        if records:
            summed_precision += rel_count / len(records)
    # Divide by the total number of relevant records,
    # plus one for the sentinel.
    return summed_precision / (records.T_q + 1)


def determine_E0(retlists, k, quantile, ascending):
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
    except InputValueError:
        # Re-raise with a proper message (the callee didn't have all info).
        raise InputValueError(
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
    raise InputValueError  # construct the message in the caller


def determine_monotonicity(retlists):
    '''
    Find out if we have ascending E values or descending scores.

    Returns True for ascending, False for descending values.
    If the monotonicity cannot be determined, raise InputValueError.
    '''
    for records in retlists:
        comp = None
        for _, score in records:
            try:
                if score > comp:
                    return True
                if score < comp:
                    return False
            except TypeError:
                comp = score
    raise InputValueError(
        'Every retrieval list was empty or repeated the same score or E-value,\n'
        'in which case you must specify through the -m/--monotonicity option\n'
        'whether the lists are ascending or descending.')


def parserecords(stream, unweighted=False):
    '''
    Iterate over RetrievalList instances parsed from plain-text.
    '''
    current = None
    stream = enumerate(stream)  # make sure we can jump ahead inside the loop
    for i, line in stream:
        try:
            current.add(line, i)
        except AttributeError:
            # current is None: Start a new retrieval list.
            if line.strip():
                # Consume the next line as well (without its line number).
                current = RetrievalList.incremental_factory(
                    line, next(stream)[1], i, unweighted)
        except BlankLineSignal:
            # This retrieval list has ended.
            # Yield it and reset current (so it won't be yielded again,
            # since multiple blank lines are allowed).
            yield current
            current = None
    if current is not None:
        # The last list might still need to be yielded.
        yield current


class RetrievalList:
    '''
    A list of rated records and some metadata.
    '''
    __slots__ = ('query', 'weight', 'T_q', 'records')

    def __init__(self, query, T_q, records, weight=1.0):
        self.query = query
        self.weight = weight
        self.T_q = T_q
        self.records = records

    @classmethod
    def incremental_factory(cls, line1, line2, no, unweighted=False):
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
                    'should have at most 2 columns.', no, line1)
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
                    'should be a positive weight.', no, line1)
        try:
            T_q = int(line2)
            if T_q < 0:
                raise ValueError()
        except ValueError:
            raise InputFormatError(
                'The line containing the number of relevant records '
                'should be a non-negative integer', no+1, line2)
        # Construct a RetrievalList with a (yet) empty records list.
        if unweighted:
            weight = 1.0
        return cls(query, T_q, [], weight)

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
                    no, line)
        self.records.append((bool(relevance), score))

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


class BlankLineSignal(Exception):
    'Empty input line.'

class InputFormatError(Exception):
    'Input text could not be parsed.'
    def __init__(self, message, line_number, line):
        super().__init__(message, line_number, line)
        self.message = message
        self.line_number = line_number
        self.line = line

    def __str__(self):
        no = 'Offending input line: {}'.format(self.line_number)
        return '\n'.join((self.message, no, self.line))

class InputValueError(Exception):
    'Input is incomplete or inconsistent.'


def zerotoone(expr):
    '''
    Make sure expr is a float in the interval ]0..1].
    '''
    q = float(expr)
    if q <= 0 or q > 1:
        raise argparse.ArgumentError(expr, 'violation of 0 < q <= 1')
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
