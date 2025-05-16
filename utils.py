#Copyright (2025) Bytedance Ltd. 
import multiprocessing as mp 
from typing import Optional
import unicodedata
from abc import ABCMeta, abstractmethod
from math import isinf, isnan
import numpy as np
import re


def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def process_answer(answer):
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def hit_calculate(result, dataset_name, anls_threshold=0.5):
    if listinstr(['TextVQA'], dataset_name):
        return [np.mean(x['match']) for x in result]
    elif listinstr(['DocVQA', 'InfoVQA'], dataset_name):
        return [0.0 if 1 - np.min(x['match']) < anls_threshold else 1 - np.min(x['match']) for x in result]
    elif listinstr(['ChartQA', 'OCRVQA'], dataset_name):
        return [np.max(x['match']) for x in result]
    else:  # default using vqa_score to calculate score
        return [np.mean(x['match']) for x in result]

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()



def process_line_WildDoc(line, method='vqa_score'):
    ret = {'index':line["index"]}
    if isinstance(line['answer'], list):
        answers = eval(line['answer'])
    else:
        answers = [line['answer']]
    if method == 'vqa_score':
        ret['gt'] = [process_answer(x) for x in answers]
        ret['pred'] = process_answer(line['prediction'])
        ret['match'] = []
        for current_idx, gtAnsDatum in enumerate(ret['gt']):
            otherGTAns = [
                item for ret_gt_idx, item in enumerate(ret['gt'])
                if ret_gt_idx != current_idx
            ]
            matchingAns = [
                item for item in otherGTAns if item == ret['pred']
            ]
            acc = min(1, float(len(matchingAns)) / 3)
            ret['match'].append(acc)
    elif method == 'anls':
        ret['gt'] = answers
        ret['pred'] = line['prediction']
        # import pdb
        # pdb.set_trace()
        ret['match'] = [anls_compute(x, ret['pred']) for x in ret['gt']]
    elif method == 'relaxed_accuracy':
        ret['gt'] = answers
        ret['pred'] = line['prediction'].strip()
        ret['match'] = [relaxed_correctness(ret['pred'], x) for x in ret['gt']]
    elif method == 'accuracy':
        ret['gt'] = answers
        ret['pred'] = line['prediction'].strip()
        ret['match'] = [(1.0 if (x.strip().lower() == ret['pred'].strip().lower()) else 0.0) for x in ret['gt']]
    else:  # default using vqa_score to calculate score
        ret['gt'] = [process_answer(x) for x in answers]
        ret['pred'] = process_answer(line['prediction'])
        ret['match'] = [x == ret['pred'] for x in ret['gt']]

    return ret

def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(
        c for c in unicodedata.normalize('NFKD', x) if unicodedata.category(c) != 'Mn'
    )
    # Normalize quotes and dashes
    x = re.sub(r'[‘’´`]', "'", x)
    x = re.sub(r'[“”]', '"', x)
    x = re.sub(r'[‐‑‒–—−]', '-', x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r'((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$', '', x.strip())
        # Remove details in parenthesis
        x = re.sub(r'(?<!^)( \([^)]*\))*$', '', x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x

def fintabnet_normalize(s):
    s = normalize(s)
    remove_words = [
        'dollar', 'gallons', 'square feet', 'shares', 'mbtu',
        'mbpd', 'mbbls', 'mmbtu', 'unit', 'gwh', 'year', 'mmcf', 'mile', 'mboe'
    ]

    # Data specific filtering using regular expressions
    # Remove special characters like $, (, and )
    s = re.sub(r'[\$\(\),]', '', s)

    # Replace "dollar" with empty string if it's not part of another word
    pattern = r'\b(' + '|'.join(remove_words) + r')s?\b'
    s = re.sub(pattern, '', s, flags=re.IGNORECASE)

    # Unit conversion dictionary with regex patterns for flexibility
    unit_conversion = {
        r' \bthousand\b': 'e3',
        r' \bmillion\b': 'e6',
        r' \bbillion\b': 'e9',
        r'\bthousand\b': 'e3',
        r'\bmillion\b': 'e6',
        r'\bbillion\b': 'e9',
        r' ?%': 'e-2',
    }

    # Convert percentages to their decimal representation.
    # Applying this after unit_conversion prevents "percent" from being processed
    # in cases like "million %", which would be incorrect.
    # s = re.sub(r' ?%', 'e-2', s)
    # s_percent = re.sub(r' ?%', '', s_percent)

    s_unit_free = s

    # Iterate over unit_conversion and apply transformations
    for pattern, value in unit_conversion.items():
        s = re.sub(pattern, value, s)
        s_unit_free = re.sub(pattern, '', s_unit_free)

    # Attempt to convert to float
    try:
        return float(s), [float(s), float(s_unit_free)]
    except ValueError:
        # Return the original string and the error for debugging purposes
        return s, [s, s_unit_free]


def calculate_consistency_WildDoc(result, anls_threshold=0.5):    
    ret = 0
    consistency = {}

    for line in result:
        unique_index = "-".join([line["index"].split("-")[0], line["index"].split("-")[1], line["index"].split("-")[3]])
        dataset_name = line["index"].split("-")[0]
        if dataset_name == "DocVQA":
            score = 0.0 if 1 - np.min(line['match']) < anls_threshold else 1 - np.min(line['match'])
        elif dataset_name == "ChartQA" or dataset_name == "TableVQA":
            score = np.max(line['match'])
        if ((dataset_name=="ChartQA" or dataset_name == "TableVQA" or dataset_name=="TableVQA") and score == 1) or (dataset_name=="DocVQA" and score >0.5):
            if unique_index in consistency:
                consistency[unique_index] += 1
            else:
                consistency[unique_index] = 1

    for key, value in consistency.items():
            ret += 1 if value == 4 else 0

    return ret / (len(result) / 4) * 100

def calculate_overall_accuracy_WildDoc(result, anls_threshold=0.5):
    score = 0
    for line in result:
        benchmark_name = line["index"].split("-")[0]
        if benchmark_name == "DocVQA":
            score += 0.0 if 1 - np.min(line['match']) < anls_threshold else 1 - np.min(line['match'])
        elif benchmark_name == "ChartQA" or benchmark_name == "TableVQA":
            score += np.max(line['match'])
        else:
            score += np.mean(line['match'])
    return score / len(result) * 100

def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')


def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)

    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split('|')]

# Value Types
class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized

class StringValue(Value):
    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' + str([self.normalized])

    def __repr__(self):
        return self.__str__()

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):
    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'N({})'.format(self.amount) + str([self.normalized])

    def __repr__(self):
        return self.__str__()

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except ValueError:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except ValueError:
                return None


class DateValue(Value):
    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx',
            )
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('D(%d,%d,%d)' % (self._year, self._month, self._day)) + str(
            [self._normalized]
        )

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


class NumberValue(Value):
    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'N({})'.format(self.amount) + str([self.normalized])

    def __repr__(self):
        return self.__str__()

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except ValueError:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except ValueError:
                return None


# Value Instantiation
def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(
            set(to_value(x, y) for (x, y) in zip(original_strings, corenlp_values))
        )
    else:
        return list(set(to_value(x) for x in original_strings))

# Check the Predicted Denotations
def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True
