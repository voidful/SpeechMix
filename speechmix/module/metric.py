import editdistance as ed


def cer_cal(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p.lower(), t.lower()))
        tot += len(t)
    return err / tot


def wer_cal(groundtruth, hypothesis):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.lower().split(' ')
        t = t.lower().split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot


