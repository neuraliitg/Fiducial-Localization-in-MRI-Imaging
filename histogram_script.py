import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()

out_arr = np.array([[1,2,4,3],[5,6,4,8]])
print(out_arr.shape)

objects = (range(len(np.ravel(out_arr))))
y_pos = np.arange(len(np.ravel(out_arr)))
performance = np.ravel(out_arr)
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Value')
plt.title('Window Value')
plt.savefig('Bargraph.png')