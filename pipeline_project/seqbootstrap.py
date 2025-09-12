import pandas as pd
import numpy as np

def getIndMatrix(barIx, t1):
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_val) in enumerate(t1.items()):
        indM.loc[t0:t1_val, i] = 1.
    return indM

def getAvgUniqueness(indM):
    c = indM.sum(axis=1)
    u = indM.div(c, axis=0)
    avgU = u[u > 0].mean()
    return avgU

def seqBootstrap(indM, sLength=None):
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series(dtype=float)
        for i in indM.columns:
            indM_ = indM[phi + [i]]
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()
        chosen = np.random.choice(indM.columns, p=prob)
        phi.append(chosen)
    return phi