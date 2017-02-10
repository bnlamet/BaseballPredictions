from players import *
from multiprocessing import Pool,freeze_support

exec(open('load.py').read())

build(ab_train, p_train)
print('Model Constructed')
evaluate(logistic, ab_test)

