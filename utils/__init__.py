def secToSample(sec):
    return int(16000 * sec)

def secToFrames(sec):
    return secToSample(sec)//160+1

def framesToSample(fr):
    return (fr-1)*160

