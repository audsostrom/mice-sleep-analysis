from inputMassager import *

#Grabs one dataframe and prints the pandas df and the tensor

test = inputMassager()
#filepath = test.askForInput("Blah!")
filepath = "CHDCtrl1_CHD801FR_normal/CHD801FR_20221123_normal.txt"
# print("getting df")
# df = test.getDfFromText()
# print(df)
print("getting tensor")
ten = test.makePeriodFromTxt(filepath, 200, 2000)
#def makePeriodFromTxt(self, filepath, periodSize, maxPeriods=None):
cols = find_time_labels(ten, "CHDCtrl1_CHD801FR_normal/CHD801FR_20221123_normal_annotated.txt")
ten1 = label_dataframe(ten, cols)
print(ten1[1701:1710])

# 1702: 3, 1703: 2, 1704: 2, 1705: 2, 1706: 2, 1707: 2, 1708: 3, 1709: 3