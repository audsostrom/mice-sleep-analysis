import tkinter as tk #tk for file dialog (requires Jinja2!!!)
from tkinter import filedialog #tkinter for file dialog

import re #regex for parsing
import numpy as np
import pandas as pd
import os
from os.path import exists
import torch

def isTextFile(filepath):
	fLen = len(filepath)
	#Check that our filepath is a .txt
	return bool((fLen > 4) and (filepath[fLen-4:] == ".txt"))


# Class that handles user input
# It holds a reference to the filepath, and generated dataframes or tensors on demand
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

		#Check that our filepath is a .txt
		if (isTextFile(self.filepath)):

			#open data for reading
			dataFp = open(self.filepath, "r")

			readingData = False
			col1, col2, col3 = [], [], []
			for line in dataFp.readlines():

				if readingData:
					#split the line by tabs and add the data to our column lists
					data_list = line.split("\t")
					col1.append(data_list[1])
					col2.append(data_list[2])
					col3.append(data_list[3])

				#Ignore initial lines
				else:
					stripped = re.sub("\s", "", line)
					#This line is the last line before data comes
					if stripped == "(m:s.ms)(mV)(mV)":
						readingData = True


			dataFp.close()

			df = pd.DataFrame(list(zip(col1, col2, col3)), columns =['Time', 'Channel1', "Channel2"])
			return df

		else:
			print("Cannot handle file", self.filepath)
			return None


	def getTensorFromText(self, maxSize=None):
		#Check that our filepath is a .txt")
		if (isTextFile(self.filepath)):

			#open data for reading
			dataFp = open(self.filepath, "r")

			readingData = False
			col1, col2, col3,  col4 = [], [], [], []

			line_count = 0
			for line in dataFp.readlines():
				if ((maxSize != None) and (maxSize <= line_count)):
					break

				if readingData:
					#split the line by tabs and add the data to our column lists
					data_list = line.split("\t")
					stripped_time = re.sub("\s", "",data_list[1])
					time = stripped_time.split(":")

					col1.append(int(time[0]))
					col2.append(np.float16(time[1]))
					col3.append(np.float16(re.sub("\s", "",data_list[2])))
					col4.append(np.float16(re.sub("\s", "",data_list[3])))

					line_count += 1

				#Ignore initial lines
				else:
					
					stripped = re.sub("\s", "", line)
					#This line is the last line before data comes
					if stripped == "(m:s.ms)(mV)(mV)":
						readingData = True

			dataFp.close()
			ten = torch.tensor([col1, col2, col3, col4],  dtype=torch.float16)
			return ten
		else:
			print("Cannot handle file", self.filepath)
			return None			

