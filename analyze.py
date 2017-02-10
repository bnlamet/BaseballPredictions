from players import *
from multiprocessing import Pool,freeze_support

exec(open('load.py').read())

if __name__ == '__main__':
	freeze_support()
	
	par_build(ab_train, p_train)
	print('Model Constructed')
	functions = [simple2, simple3, simple4, logistic, weight1, markov2, markov3, markov4, bayes, forest]
	p = Pool()

	for function in functions:
		par_eval(p, function, ab_test)
#		evaluate(function, ab_test)

