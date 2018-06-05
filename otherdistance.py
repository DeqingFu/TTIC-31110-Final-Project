import editdistance

def _toWER(process, results):
    '''
    Computes WER according to the equal function
    --------------------------------------------
    input: rocess function and results
    output: WER
    '''
    dist = 0.
    for label, pred in results:
        dist += editdistance.eval(list(map(lambda x : process(x), label)), list(map(lambda x : process(x), pred)))
    total = sum(len(label) for label, _ in results)
    return dist / total, dist 


def _noTone(s1):
    return ''.join(list(map(lambda x : x if x.islower() else '', s1)))


def _pingQiaoShe(s1):
    tmp1 = ''.join(list(map(lambda x : x if x.islower() else '', s1)))
    fromLst = ['sh', 'zh', 'ch']
    toLst = ['s', 'z', 'c']
    for i in range(len(fromLst)):
        tmp1 = tmp1.replace(fromLst[i], toLst[i])
    return tmp1


def _qianHouBi(s1):
    tmp1 = ''.join(list(map(lambda x : x if x.islower() else '', s1)))
    fromLst = ['ang', 'eng', 'ing']
    toLst = ['an', 'en', 'in']
    for i in range(len(fromLst)):
        tmp1 = tmp1.replace(fromLst[i], toLst[i])
    return tmp1


def _moHu(s1):
    tmp1 = ''.join(list(map(lambda x : x if x.islower() else '', s1)))
    fromLst = ['sh', 'zh', 'ch', 'ang', 'eng', 'ing']
    toLst = ['s', 'z', 'c', 'an', 'en', 'in']
    for i in range(len(fromLst)):
        tmp1 = tmp1.replace(fromLst[i], toLst[i])
    return tmp1


def compute_wer_notone(results):
    return _toWER(_noTone, results)


def compute_wer_pingqiaoshe(results):
    return _toWER(_pingQiaoShe, results)


def compute_wer_qianhoubi(results):
    return _toWER(_qianHouBi, results)


def compute_wer_mohu(results):
    return _toWER(_moHu, results)
