"""Log pre-processing: header splitting, identifier extraction and variable masking.

Masking modes
-------------
- ``none``    : no masking, Drain sees the raw Content.
- ``regex``   : logpai-style masking, each regex in a list is replaced by ``<*>``
                (used for the loghub benchmark regexes and for the paper's regexes).
- ``qin``     : the general preprocessing framework of Qin et al. (SANER 2025),
                vendored from https://github.com/mooselab/Preprocessing-is-All-You-Need
                (benchmark/logparser/utils/preprocessing.py). Includes its
                "estimation stage": the first 2000 lines are masked with the full
                regex sequence and the set of regex types that actually matched is
                then used for the rest of the file (exactly as in their Drain.py).

Identifiers are always extracted from the *raw* Content, before masking.
"""
import re

# --------------------------------------------------------------------------
# Qin et al. (SANER 2025) preprocessing framework -- vendored verbatim
# --------------------------------------------------------------------------
weekday_abb = ['Mon', 'Tue', 'Wed', "Thu", 'Fri', 'Sat', 'Sun']
weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_abb = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

QIN_REGEX = {
    'IPv4_port': r'(/|)(\d+\.){3}\d+(:\d+)?',
    'host_port': r'([\w-]+\.)+[\w-]+\:\d+',
    'package_host': r'([\w-]+\.){2,}[\w-]+(\$[\w-]+)*(\@[\w-]+)?',
    'Mac_address': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
    'IPv6': r'(([0-9a-fA-F]{1,4}:){7}([0-9a-fA-F]{1,4}|:)|(([0-9a-fA-F]{1,4}:){1,7}|:):((:[0-9a-fA-F]{1,4}){1,7}|:))',
    'path': r'(/|)(([\w.-]+|\<\*\>)/)+([\w.-]+|\<\*\>)',
    'size': r'\b\d+\.?\d*\s?([KGTMkgtm]?(B|b)|([KGTMkgtm]))\b',
    'duration': r'\b\<?\d+\s?(sec|s|ms)\b',
    'block': r'blk\_\-?\d+',
    'numerical': r'\b(\-?\+?\d+\.?\d*)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b',
    'time': r'\b\d{2}:\d{2}(:\d{2}|:\d{2},\d+)?\b',
    'date': r'\b(\d{4}-\d{2}-\d{2})|\d{4}/\d{2}/\d{2}\b',
    'url': r'\bhttps?:\/\/(www\.)?[a-zA-Z0-9-]+(\.[a-zA-Z]{2,})+(:[0-9]{1,5})?(\/[^\s]*)?\b',
    'weekday_months': r'\b(%s)\b' % '|'.join(weekday_abb + weekday + month_abb + months),
}
QIN_SEQUENCE = ['url', 'IPv4_port', 'host_port', 'package_host', 'IPv6', 'Mac_address', 'time',
                'path', 'block', 'date', 'duration', 'size', 'numerical', 'weekday_months']
QIN_ESTIMATION_LINES = 2000
_QIN_COMPILED = {k: re.compile(v) for k, v in QIN_REGEX.items()}
_REPEAT_STAR = re.compile(r'(\<\*\>\s?)+\<\*\>')


def qin_preprocess(log, estimation_stage=False, use_sequence=QIN_SEQUENCE):
    """Port of Qin et al.'s ``preprocess`` (use_keys=False branch, i.e. every mask is <*>)."""
    matched_types = []
    new_log = log
    for key in use_sequence:
        new_log = _QIN_COMPILED[key].sub('<*>', new_log)
        if estimation_stage and new_log != log:
            matched_types.append(key)
            log = new_log
    new_log = _REPEAT_STAR.sub('<*>', new_log)
    if estimation_stage:
        return new_log, matched_types
    return new_log


# --------------------------------------------------------------------------
# Masker: a picklable object used by the (multiprocess) parser
# --------------------------------------------------------------------------
class Masker:
    def __init__(self, mode, regexes=None, narrow=True):
        """narrow=False (config 'qin_full', sensitivity only): skip Qin's estimation-stage narrowing
        and apply the full regex sequence to every line."""
        assert mode in ("none", "regex", "qin"), mode
        self.mode = mode
        self.narrow = narrow
        self.regexes = list(regexes or [])
        self._compiled = [re.compile(r) for r in self.regexes]
        self.qin_sequence = list(QIN_SEQUENCE)  # narrowed after estimation

    def estimate(self, first_contents):
        """Qin estimation stage. Returns the masked first lines; narrows the regex sequence."""
        if self.mode != "qin":
            return [self.mask(c) for c in first_contents]
        out, matched = [], set()
        for c in first_contents:
            m, types = qin_preprocess(c, estimation_stage=True)
            out.append(m)
            matched.update(types)
        if self.narrow:
            self.qin_sequence = [k for k in QIN_SEQUENCE if k in matched]
        return out

    def mask(self, content):
        if self.mode == "none":
            return content
        if self.mode == "regex":
            for r in self._compiled:
                content = r.sub('<*>', content)
            return content
        return qin_preprocess(content, estimation_stage=False, use_sequence=self.qin_sequence)

    def describe(self):
        if self.mode == "qin":
            return {"mode": "qin" if self.narrow else "qin_full", "sequence_after_estimation": self.qin_sequence}
        return {"mode": self.mode, "regexes": self.regexes}


# --------------------------------------------------------------------------
# Header splitting (logpai generate_logformat_regex) and identifiers
# --------------------------------------------------------------------------
def logformat_regex(log_format):
    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)
    regex = ''
    for k, s in enumerate(splitters):
        if k % 2 == 0:
            regex += re.sub(' +', r'\\s+', s)
        else:
            header = s.strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    return headers, re.compile('^' + regex + '$')


BLOCK_RE = re.compile(r'blk_-?\d+')


def hdfs_block_ids(content):
    """All block ids referenced by a line, de-duplicated, in order (loglizer uses a set)."""
    return list(dict.fromkeys(BLOCK_RE.findall(content)))
