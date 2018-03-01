from fnmatch import fnmatch
import re

class Entry(object):
    def __init__(self, operator, patterns):
        self.operator = operator
        self.patterns = self.make_patterns(patterns)
        self.str_patterns = patterns

    def make_patterns(self, patterns):
        d = []
        for arg, p in patterns.items():
            if isinstance(p, type):
                # special case if pattern is already an exact type
                # treat it like a string.
                p = p.__name__

            d.append((arg, re.compile(translate(p))))
        return dict(d)

    def match(self, operands):
        """ See if the entry matches the set of operands.

            Extra arguments in the operands are ignored.
        """

        # if all args in the pattern are supplied?
        if any(arg not in operands for arg in self.patterns):
            return None

        score_major = 0 # number of glob groups
        score_minor = 0 # length of glob pattern matches

        typenames = _gettypenames(operands)

        for arg in self.patterns:
            pattern = self.patterns[arg]
            typename = typenames[arg]

            r = pattern.match(typename)
            if not r:
                return None

            score_major += len(r.groups())
            score_minor += sum([len(x) for x in r.groups()])

        # prefer more specific rules
        return len(operands) - len(self.patterns), score_major, score_minor

    def __repr__(self):
        return "Entry: %s : %s" % (self.operator, self.str_patterns)

class Registry(object):
    """ A registory of operators and matching rules """
    def __init__(self):
        self._all = {}

    def match_all(self, name, operands):
        """ Find by name the operator entry that declares it can work on these operands """
        desc = self._all.get(name, [])
        r = []
        for i, entry in enumerate(desc[::-1]):
            score = entry.match(operands)
            if score is not None:
                r.append((score, i, entry, _gettypenames(operands)))

        r = sorted(r)
        return r

    def match(self, operator, operands, name=None):
        """ Find optimal operator entry that declares it can work on these operands """
        if name is None:
            name = _getname(operator)

        r = self.match_all(name, operands)

        if len(r) == 0:
            raise KeyError('no operator is found for %s' % str(operator))

        score, i, entry, types = r[0]
        return entry.operator

    def register(self, operator, patterns, name=None):
        """ Regsitor an operator entry that declares it can work on these operands matched by the pattern. 

            patterns: a dict of arg:fnmatch patterns -- must match from beginning to end on the name of the type;
        """
        if name is None:
            name = _getname(operator)

        desc = self._all.get(name, [])

        if patterns not in desc:
            desc.append(Entry(operator, patterns))

        self._all[name] = desc

def _gettypenames(operands):
    """ helper function to obtain type name dict from an input dict of operands """
    return dict([(arg, type(operand).__name__) for arg, operand in operands.items()])

def _getname(operator):
    """ helper function to infer operator name """
    if isinstance(operator, str):
        return operator
    return operator.__name__

def translate(pat):
    """Translate a shell PATTERN to a regular expression.
    There is no way to quote meta-characters.

    This differs from the one from fnmatch in that it converts each
    glob as a group. Such that we can count the number of chars that matched.

    """

    i, n = 0, len(pat)
    res = ''
    while i < n:
        c = pat[i]
        i = i+1
        if c == '*':
            res = res + r'(.*)'
        elif c == '?':
            res = res + r'(.)'
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j+1
            if j < n and pat[j] == ']':
                j = j+1
            while j < n and pat[j] != ']':
                j = j+1
            if j >= n:
                res = res + '\\['
            else:
                stuff = pat[i:j].replace('\\','\\\\')
                i = j+1
                if stuff[0] == '!':
                    stuff = '^' + stuff[1:]
                elif stuff[0] == '^':
                    stuff = '\\' + stuff
                res = r'%s([%s])' % (res, stuff)
        else:
            res = res + re.escape(c)
    return res + '\Z(?ms)'
