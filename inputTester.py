from adam import *

#Grabs one dataframe and prints the pandas df and the tensor

test = inputMassager()
test.askForInput("Blah!")
# print("getting df")
# df = test.getDfFromText()
# print(df)
print("getting tensor")
ten = test.getTensorFromText()
print(ten)