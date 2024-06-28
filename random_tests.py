myList = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
minValue = min(myList)

myList = [x - minValue for x in myList]
print(myList)