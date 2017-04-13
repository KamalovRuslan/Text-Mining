import os, sys, codecs

s = set([1,2,3,4,5,6])
ss  = set([7,8,9])
b = {x : 1./len(ss) for x in ss}
a = {str(x): b for x in s}
print a
