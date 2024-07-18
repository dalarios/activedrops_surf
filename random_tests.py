import numpy as np

myList = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
minValue = min(myList)

myList = [x - minValue for x in myList]
print(myList)

myArray = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
myArray = np.array([])
print(myArray)