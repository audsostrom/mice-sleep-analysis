import tkinter as tk #tk for file dialog (requires Jinja2!!!)
from tkinter import filedialog #tkinter for file dialog

import re #regex for parsing
import numpy as np
import pandas as pd
import os
from os.path import exists

root = tk.Tk()
root.withdraw() #Hide root window

#HARDCODED FILEPATH

#Filepath Determined by tk dialog
filepath = filedialog.askopenfilename()


if not (os.path.exists(filepath)):
	print("Cannot find file at path:", filepath)


fLen = len(filepath)
if (fLen > 4) and (filepath[fLen-4:] == ".txt"):
	data = open(filepath, "r")    # note mode "r"
	readingData = False
	col1 = []
	col2 = []
	col3 = []
	for line in data.readlines():
		if readingData:
			data_list = line.split("\t")
			# print(
			# 	re.sub("\s", "", data_list[1]),
			# 	re.sub("\s", "", data_list[2]),
			# 	re.sub("\s", "", data_list[3]),
			# 	)
			col1.append(data_list[1])
			col2.append(data_list[2])
			col3.append(data_list[3])
		else:
			stripped = re.sub("\s", "", line)
			if stripped == "(m:s.ms)(mV)(mV)":
				readingData = True


	data.close()
else:
	print("Cannot handle file", filepath)
	exit()
df = pd.DataFrame(list(zip(col1, col2, col3)), columns =['Time', 'Channel1', "Channel2"])
print(df)