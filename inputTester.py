from inputMassager import *
import numpy as np

#Grabs one dataframe and prints the pandas df and the tensor

massager = inputMassager()

#massager = massager.askForInput("Blah!")

filepath = R"C:\Users\Adam\Desktop\CHDCtrl1_CHD801FR_normal\CHD801FR_20221123_normal.txt"

print("getting tensor")



intermediate = massager.makePeriodFromTxt(filepath, 200)
#def makePeriodFromTxt(self, filepath, periodSize, maxPeriods=None):

cols = find_time_labels(intermediate, R"C:\Users\Adam\Desktop\CHDCtrl1_CHD801FR_normal\CHD801FR_20221123_normal_annotated.txt")
output = label_dataframe_new(intermediate, cols)


print("output of out new labeling, length:", len(output))

one_c, two_c, three_c, four_c = 0, 0, 0, 0
for label in output:
    match str(label):
        case "1":
            one_c += 1
            break
        case "2":
            two_c += 1
            break
        case "3":
            three_c += 1
            break
        case "4":
            four_c += 1
            break
        case _:
            print("broken", label)
            break
print("total output breakdown:",
      "number of 1s:", one_c, "\n",
      "number of 2s:", two_c, "\n",
      "number of 3s:", three_c, "\n",
      "number of 4s:", four_c, "\n",
      )

one_c, two_c, three_c = 0.0, 0.0, 0.0
one_list, two_list, three_list = [], [], []
for i in range (0, len(cols[0])):
    prev_i = i-1 if i>0 else 0
    if cols[1][i] == 1:
        one_c += (cols[0][i] - cols[0][prev_i])
        one_list.append(cols[0][i] - cols[0][prev_i])
    elif cols[1][i] == 2:
        two_c += (cols[0][i] - cols[0][prev_i])
        two_list.append(cols[0][i] - cols[0][prev_i])
    elif cols[1][i] == 3:
        three_c += (cols[0][i] - cols[0][prev_i])
        three_list.append(cols[0][i] - cols[0][prev_i])

print("time spend in 1:", one_c, "mean time in", np.mean(one_list))
print("time spend in 2:", two_c, "mean time in", np.mean(two_list))
print("time spend in 3:", three_c, "mean time in", np.mean(three_list))
# 1702: 3, 1703: 2, 1704: 2, 1705: 2, 1706: 2, 1707: 2, 1708: 3, 1709: 3