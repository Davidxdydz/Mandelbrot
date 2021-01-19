import time

def timer(f,*args,**kwargs):
    start = time.time()
    result = f(*args,**kwargs)
    end = time.time()
    return result, end-start