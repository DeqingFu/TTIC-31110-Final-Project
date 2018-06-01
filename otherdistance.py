def _toWER(eqFunction, results):
    '''
    Computes WER according to the equal function
    --------------------------------------------
    input: equal function and results
    output: WER
    '''
    dist = 0.
    for label, pred in results:
        dist += _levenshtein(label, pred, eqFunction)
    total = sum(len(label) for label, _ in results)
    return dist / total


def _levenshtein(s1, s2, eqFunction):
    '''
    Levenshtein distance credit to Wikipedia
    ----------------------------------------
    input: two strings and a equal function
    output: Levenshtein distance
    '''
    if len(s1) < len(s2):
        return _levenshtein(s2, s1, eqFunction)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if eqFunction(c1, c2) else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _eqNoTone(c1, c2):
    tmp1 = ''.join(list(map(lambda x : x if x.islower() else '', c1)))
    tmp2 = ''.join(list(map(lambda x : x if x.islower() else '', c2)))
    return tmp1 == tmp2


def _eqPingQiaoShe(c1, c2):
    tmp1 = c1
    tmp2 = c2
    fromLst = ['sh', 'zh', 'ch']
    toLst = ['s', 'z', 'c']
    for i in range(3):
        tmp1 = tmp1.replace(fromLst[i], toLst[i])
        tmp2 = tmp2.replace(fromLst[i], toLst[i])
    return tmp1 == tmp2


def _eqQianHouBi(c1, c2):
    tmp1 = c1
    tmp2 = c2
    fromLst = ['ang', 'eng', 'ing']
    toLst = ['an', 'en', 'in']
    for i in range(3):
        tmp1 = tmp1.replace(fromLst[i], toLst[i])
        tmp2 = tmp2.replace(fromLst[i], toLst[i])
    return tmp1 == tmp2


def compute_wer_notone(results):
    return _toWER(_eqNoTone, results)


def compute_wer_pingqiaoshe(results):
    return _toWER(_eqPingQiaoShe, results)


def compute_wer_qianhoubi(results):
    return _toWER(_eqQianHouBi, results)