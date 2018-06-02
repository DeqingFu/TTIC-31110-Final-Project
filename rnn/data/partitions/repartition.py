#!/usr/bin/env python3
import random

def countLine(file):
	'''
	Counts the number of lines in a file
	------------------------------------
	input: a string indicating location of file
	output: an int
	'''
	with open(file, 'r') as f:
		for i, _ in enumerate(f):
			pass
	return i + 1


def assign(numL):
	'''
	Assign lines to train, dev, and test w/z ratio 8:1:1
	----------------------------------------------------
	inputs: line number
	output: an assignment string of (line number, assignment);
			0 - train, 1 - dev, 2 - test
	'''
	random.seed()

	assigned = [False] * numL
	numAssgn = 0
	res = [-1] * numL

	def assignTo(which):
		nonlocal numAssgn
		nonlocal numL
		nonlocal assigned
		nonlocal res
		if numAssgn == numL:
			return
		flg = True
		while flg:
			i = random.randrange(numL)
			if not assigned[i]:
				flg = False
				assigned[i] = True
				numAssgn += 1
				res[i] = which

	while True:
		for _ in range(8):
			assignTo(0)
		assignTo(1)
		assignTo(2)
		if numAssgn == numL:
			break

	return res


def writeFiles(file, assgnmnt):
	'''
	Writes files according to assignment
	------------------------------------
	input: file and assignment string
	output: void
	'''
	with open(file, 'r') as whole, open('train.json', 'w') as train, \
	     open('dev.json', 'w') as dev, open('test.json', 'w') as test:
		for i, l in enumerate(whole):
			if assgnmnt[i] == 0:
				train.write(l)
			elif assgnmnt[i] == 1:
				dev.write(l)
			elif assgnmnt[i] == 2:
				test.write(l)
			else:
				print("ERROR: bad assignment")
				exit(1)


if __name__ == '__main__':
	numL = countLine('data.json')
	assgnmnt = assign(numL)
	writeFiles('data.json', assgnmnt)