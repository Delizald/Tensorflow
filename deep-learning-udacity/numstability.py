#Testing numerical stability
a = 1000000000
for i in xrange(1000000000):
	a = a + 1e-6
print a - 1000000000

#Ideally we need:
#O Mean
# = Variance