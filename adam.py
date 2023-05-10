import tkinter as tk #tk for file dialog (requires Jinja2!!!)
from tkinter import filedialog #tkinter for file dialog

import re #regex for parsing
import numpy as np
import pandas as pd
import os
from os.path import exists


# Class that handles user input
class inputMassager():

	def __init__(self):

		self.filepath = ""

	# Opens a file selection dialog with optional message
	# Returns True on valid filepath selection, False otherwise
	def askForInput(self, title_msg = "Select Data File"):
		root = tk.Tk()  #init tkinter root
		root.withdraw() #Hide root window

		#Filepath Determined by tk dialog
		self.filepath = filedialog.askopenfilename(title = title_msg)


		if not (os.path.exists(self.filepath)):
			print("Cannot find file at path:", self.filepath)
			return False
		else:
			return True

	# Opens the saved DataFrame
	# Returns Df if successful, None otherwise
	def getDfFromText(self):

		fLen = len(self.filepath)

		#Check that our filepath is a .txt
		if (fLen > 4) and (self.filepath[fLen-4:] == ".txt"):

			#open data for reading
			data = open(self.filepath, "r")

			readingData = False
			col1, col2, col3 = []

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

				#Ignore initial lines
				else:
					stripped = re.sub("\s", "", line)
					#This line is the last line before data comes
					if stripped == "(m:s.ms)(mV)(mV)":
						readingData = True


			data.close()

			df = pd.DataFrame(list(zip(col1, col2, col3)), columns =['Time', 'Channel1', "Channel2"])
			# print(df)
			return df

		else:
			print("Cannot handle file", self.filepath)
			return None

test = inputMassager()
test.askForInput("Blah!")
df = test.getDfFromText()
print(df)
