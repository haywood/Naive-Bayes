#!/usr/bin/pypyFiles/bin/pypy

import random;
import sys;
import csv;

class NaiveBayes:

	def __init__(self, n):

		self.data = [];#the data
		self.dim = n;#dimension of the data
		self.values = {};#possible values
		self.totals = {};#number of times a value appears
		self.counts = [{} for i in range(n)];#number of feature value appearences by index and value
		self.possible = [set() for i in range(n)];#range possible values for each feature

	def getData(self):
		return self.data;

	def getDim(self):
		return self.dim;

	def addData(self, data):
		
		for d in data:
			if len(d[0]) != self.dim:
				return False;

		self.data += data;

		for d in data:

			if d[1] in self.values:
				self.values[d[1]] += 1;

			else: self.values[d[1]] = 1;

		return True;

	def train(self):

		for d in self.data:

			e = d[0]; v = d[1];
			
			if v in self.totals:
				self.totals[v] += 1;
			else: self.totals[v] = 1;

			for i in range(self.dim):
				
				if v not in self.counts[i]:
					self.counts[i][v] = {};

				f = e[i];
				self.possible[i].add(f);

				if f in self.counts[i][v]:
					self.counts[i][v][f] += 1;
				else: self.counts[i][v][f] = 1;

		"""		
		for v in self.values:
			dv = filter(lambda x: x[1] == v, self.data);
			den = float(len(dv));
			self.conds[v] = {};
			for d in dv:
				e = d[0];
				for f in e:
					if f in self.conds[v]:
						self.conds[v][f] += 1;
					else: self.conds[v][f] = 1;
			for f in self.conds[v]:
				self.conds[v][f] /= den;
		"""

	def predict(self, obj):
		
		if len(obj) != self.dim:
			raise ValueError('Unclassifiable object. Wrong feature space.');
		
		newData = {};
		ps = [];

		for i in range(self.dim):
			if obj[i] not in self.possible[i]:
				newData[i] = obj[i];

		for v in self.values:
			p = float(self.values[v])/len(self.data);
			i = 0;
			while p and i < self.dim:
				if i not in newData:
					f = obj[i];
					if f in self.counts[i][v]:
						p *= float(self.counts[i][v][f])/self.totals[v];
					else:
						p = 0.0;
				i += 1;
			"""
		for v in self.values:
			p = self.values[v];
			for f in obj: 
				if f in self.counts[v]:
					p *= self.counts[v][f];
				else:
					p = 0.0;
					break;
			"""
			ps.append((p, v));

		return max(ps, key=lambda x: x[0]), newData;

def test():
		
	sheetReader = csv.reader(open('training.csv', 'r'));
	a = sheetReader.next();
	examples = [];
	possible = [ set() for i in range(len(a)) ];

	print "\nTraining...\n"
	for example in sheetReader:
		for i in range(len(example)):
			example[i] = example[i].strip();
			possible[i].add(example[i]);
		v = example.pop();
		if example[0]:
			examples.append((example, v));
			print example, v;

	tests = [];
	print "\nTesting...\n";
	sheetReader = csv.reader(open('testing.csv', 'r'));
	for row in sheetReader:
		for i in range(len(row)):
			row[i] = row[i].strip();
		v = row.pop();
		if row[0]:
			tests.append((row, v));
			print row, v;


	nb = NaiveBayes(len(examples[0][0]));
	nb.addData(examples);
	nb.train();

	d = float(len(examples));
	n = 0;

	for e in tests:

		#unseen = [random.choice(list(possible[i])) for i in range(len(possible))];
		#v = unseen.pop(10);
		v = e[1];
		test = nb.predict(e[0]);
		if test[0][1] == v: n += 1;

	print n/d, n, d;

	#unseen = [random.choice(list(possible[i])) for i in range(len(possible))];
	#v = unseen.pop(10);
	#print nb.predict(unseen), v, unseen;
	#for h in nb.counts:
		#print h;


if __name__ == '__main__':
	test();
