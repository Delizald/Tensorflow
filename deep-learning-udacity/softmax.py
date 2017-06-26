########Creating  a softmax function
#sudo apt-get install python-matplotlib
import numpy as np

test_data=[2.0,1.0,0.5]

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x),axis=0)

print (softmax(test_data))

import matplotlib.pyplot as plt

x = np.arange(-2,6.0,0.1)
scores = np.vstack([x,np.ones_like(x),0.2 * np.ones_like(x)])

plt.plot(x,softmax(scores).T, linewidth=2)
plt.show()

#if you multiply the scores by 10, the scores either get very large or very small
#print (softmax(test_data * 10))