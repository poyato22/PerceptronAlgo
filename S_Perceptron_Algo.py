# Imports
import numpy as np
import matplotlib.pyplot as plt

# Read the data from our file into xs and ys
x = []
y = []

# Open the dataset and read the data
with open("separable.csv", "r") as f:
  for row in f:
    if "x1" in row:
      continue
    else:
      rdata = row.split(",")
      # Get the x1 and x2 coordinates for the x-vector
      # Lift the data by appending a third x3=1 value to the x-vector
      x.append([float(rdata[0]), float(rdata[1]), 1])
      # Get the y label
      y.append(int(rdata[2]))

# Convert our lists into NumPy arrays
x = np.array(x)
y = np.array(y)

# Plot the data
for idx, xi in enumerate(x):
  if y[idx] > 0:
    plt.scatter(xi[0], xi[1], c='b')
  else:
    plt.scatter(xi[0], xi[1], c='r')

# Weights
w = [0.0, 0.0, 0.0]

# Returns what point is classifed wrong
def classified_wrong(data, label, seperator):
    for i in range(len(data)):
        yi = label[i]
        xi = data[i]
        if (np.dot(seperator, xi)) * yi <= 0:
            return i
    return -1

# Updates the graph
while classified_wrong(x,y,w) != -1:

    i = classified_wrong(x,y,w)
    
    # Updates W
    w += (x[i] * y[i])
    print(w)

    plt.clf()

    # Plot the datapoints
    for idx, xi in enumerate(x):
        if y[idx] > 0:
          plt.scatter(xi[0], xi[1], c='b')
        else:
          plt.scatter(xi[0], xi[1], c='r')

    # Plot the decision boundary
    w_x = np.linspace(0, 10)
    # This line below converts the decision boundary w into y=mx+b form
    plt.plot(w_x, -w[0]/w[1] * w_x + -w[2]/w[1], c='k')

    # Scaling the axes
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.show(block = False)

    plt.pause(0.01)

print("\nFinal W:")
print(w)
plt.show()



