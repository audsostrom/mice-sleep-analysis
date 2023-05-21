from inputMassager import *

#Grabs one dataframe and prints the pandas df and the tensor

test = inputMassager()
#filepath = test.askForInput("Blah!")
filepath = R"\Users\sreye\PycharmProjects\pythonProject\CheckMice\CHDCtrl1_CHD801FR_normal\CHD801FR_20221123_normal.txt"
# print("getting df")
# df = test.getDfFromText()
# print(df)
print("getting tensor")
ten = test.makePeriodFromTxt(filepath, 200, 50)
#def makePeriodFromTxt(self, filepath, periodSize, maxPeriods=None):
print(ten)