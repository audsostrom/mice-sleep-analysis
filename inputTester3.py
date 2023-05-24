from inputMassager import *
fp = R"C:\Users\0feig\Desktop\CHDCtrl1_CHD801FR_normal\CHD801FR_20221123_normal.txt"
annotated_fp = R"C:\Users\0feig\Desktop\CHDCtrl1_CHD801FR_normal\CHD801FR_20221123_normal_annotated.txt"

labels, c1, c2 = get_labeled_data(fp, annotated_fp, 500)

print(labels.shape)
print(c1.shape)
print(c2.shape)