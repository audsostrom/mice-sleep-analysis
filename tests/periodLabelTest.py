from inputMassager import *
import numpy as np
# =============== Summary =============== #
# This file tests our input module and it's ability to accurately label periods 
# =============== ======= =============== #

#init input massager
massager = inputMassager()

#File Dialog
#filepath = massager.askForInput("Blah!")

#Hardcoded Path
filepath = R"C:\Users\thesp\Desktop\CHDCtrl1_CHD801FR_normal\CHD801FR_20221123_normal.txt"

#Create intermediate dataframe
period_size = 100
print("Making period from txt, period size:", period_size)

intermediate = makePeriodFromTxt(filepath, period_size)

# Get all end time labels
print("finding time labels")
cols = find_time_labels(R"C:\Users\thesp\Desktop\CHDCtrl1_CHD801FR_normal\CHD801FR_20221123_normal_annotated.txt")

#Test label_dataframe_new
print("running label dataframe new")
labels, c1, c2  = label_dataframe_new(intermediate, cols, period_size)

print("\n\noutput of out new labeling, period count:", len(labels))

# Count periods in output
one_period_c, two_period_c, three_period_c, four_period_c = 0, 0, 0, 0
for label  in labels:
    match label:
        case 1:
            one_period_c += 1
        case 2:
            two_period_c += 1
        case 3:
            three_period_c += 1
        case 0:
            four_period_c += 1
        case _:
            print("broken", label)

#Compute percentages
total_c = one_period_c + two_period_c + three_period_c + four_period_c
one_percent = (one_period_c/total_c)*100 
two_percent = (two_period_c/total_c)*100 
three_percent = (three_period_c/total_c)*100 
four_percent = (four_period_c/total_c)*100

#spit out output
print("\n\nLabel_dataframe_new output summary:\n",
      "number of 1 periods:", one_period_c, " ", f"{one_percent:.2f}%","\n",
      "number of 2 periods:", two_period_c, " ", f"{two_percent:.2f}%", "\n",
      "number of 3 periods:", three_period_c, " ",  f"{three_percent:.2f}%", "\n",
      "number of 0 (ARTIFACT) periods:", four_period_c, " ",  f"{four_percent:.2f}%", "\n",
      )

#iter over all annotated labels and count time in each
one_annotated_sec, two_annotated_sec, three_annotated_sec = 0.0, 0.0, 0.0
one_list, two_list, three_list = [], [], []
for i in range (0, len(cols[0])):
    prev_i = i-1 if i>0 else 0
    if cols[1][i] == 1:
        one_annotated_sec += (cols[0][i] - cols[0][prev_i])
        one_list.append(cols[0][i] - cols[0][prev_i])
    elif cols[1][i] == 2:
        two_annotated_sec += (cols[0][i] - cols[0][prev_i])
        two_list.append(cols[0][i] - cols[0][prev_i])
    elif cols[1][i] == 3:
        three_annotated_sec += (cols[0][i] - cols[0][prev_i])
        three_list.append(cols[0][i] - cols[0][prev_i])

#compute percentages
total_annotated_c = one_annotated_sec + two_annotated_sec + three_annotated_sec
one_percent = (one_annotated_sec/total_annotated_c)*100
two_percent = (two_annotated_sec/total_annotated_c)*100
three_percent =  (three_annotated_sec/total_annotated_c)*100
avg_per = total_annotated_c/len(labels)

#spit out summary
print("\nAnnotated Data Summary:")
print("time spend in 1:", one_annotated_sec, " ",f"{one_percent:.2f}%" , "mean time:", np.mean(one_list))
print("time spend in 2:", two_annotated_sec, " ",f"{two_percent:.2f}%" , "mean time:", np.mean(two_list))
print("time spend in 3:", three_annotated_sec, " ",f"{three_percent:.2f}%","mean time:", np.mean(three_list))
print("total time in data:", f"{total_annotated_c:.2f}", "seconds")
print("avg period is:", f"{avg_per:.2f}" , "seconds")
