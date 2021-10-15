from typing import List, Union
import time as t
import numpy as np

def check_time(func):
    def decorated(iterNum: int):
        start_time = t.time()
        # print(f"Start Time : {t.localtime().tm_min}분 {t.localtime().tm_sec}초")
        func(iterNum)
        # print(f"End Time : {t.localtime().tm_min}분 {t.localtime().tm_sec}초")
        print(f"{func.__name__}의 총 걸린 시간 : {t.time()-start_time}")
    return decorated

class Experiment:
    def __init__(self, number: int):
        self.iter = number

    @check_time
    def ExperimentNumpy(self):
        counter = 0
        for _ in range(self.iter):
            counter = np.add(counter, 1)
        print(counter)

    @check_time
    def ExperimentPython3(self):
        counter = 0
        for _ in range(self.iter):
            counter += 1
        print(counter)

def main():
    test = Experiment(10000000)
    test.ExperimentPython3() # 약 0.3초
    test.ExperimentNumpy() # 약 10초

if __name__ == '__main__':
    main()