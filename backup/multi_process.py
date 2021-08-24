from multiprocessing import Pool
import os, time, random

def work(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    i_range = (1, 11)
    j_range = (1, 10)
    pool_num = (j_range[1] - j_range[0]) * (i_range[1] - i_range[0])
    p = Pool(pool_num)
    for i in range(5):
        p.apply_async(work)