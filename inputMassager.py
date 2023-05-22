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
		self.x = 0

	# Opens a file selection dialog with optional message
	# Returns filepath
	def askForInput(self, title_msg = "Select Data File"):
		root = tk.Tk()  #init tkinter root
		root.withdraw() #Hide root window

		#Filepath Determined by tk dialog
		filepath = filedialog.askopenfilename(title = title_msg)


		if not (os.path.exists(filepath)):
			print("Cannot find file at path:", filepath)
			return ""
		else:
			return filepath

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

	# Makes periods of the same size form text file
	# Takes filepath of txt file, period_size, and optional max periods.
	def makePeriodFromTxt(self, filepath, periodSize, maxPeriods=None):

		#Check that our filepath is a .txt
		if (isTextFile(filepath)):

			#open data for reading
			dataFp = open(filepath, "r")

			readingData = False
			startTime, endTime, c1, c2 = [], [], [], []

			#datapoint per period
			datapoint_count = 0
			period_count = 0
			working_c1 = []
			working_c2 = []
			count = 0
			for line in dataFp.readlines():
				#print("We are here")
				#print("count is:", count)
				count += 1
				if readingData:
					#split the line by tabs and add the data to our column lists
					data_list = line.split("\t")
					stripped_time = re.sub("\s", "",data_list[1])

					if ((maxPeriods!= None) and (period_count >= maxPeriods)):
						#print("period count is", period_count)
						#print("WE HAVE BROKE")
						break
					#print("We are now here")

					#If our datapoint count is less than period_size
					if datapoint_count < periodSize:
						#print("datapoint count", datapoint_count, "Period size", periodSize)

						if datapoint_count == 0:
							#time in seconds
							time = stripped_time.split(":")
							time = float(float(time[0]) * 60) + float(time[1])
							startTime.append(time)

						#add our new datapoints to working channels

						working_c1.append(float(data_list[2]))
						working_c2.append(float(data_list[3]))
						#print("data 2", working_c1)
						#print("data 3", working_c2)
						datapoint_count +=1

					# Add new row to our output df
					else:

						# time in seconds
						time = stripped_time.split(":")
						time = float(float(time[0]) * 60) + float(time[1])

						# making our new row
						endTime.append(time)
						assert len(working_c1) == periodSize
						assert len(working_c2) == periodSize

						c1.append(working_c1)
						c2.append(working_c2)

						# clear out working sets
						working_c1 = []
						working_c2 = []

						period_count += 1
						#print("period count is", period_count)
						datapoint_count = 0
						#print("We are also here")



				#Ignore initial lines
				else:
					stripped = re.sub("\s", "", line)
					#This line is the last line before data comes
					if stripped == "(m:s.ms)(mV)(mV)":
						readingData = True


			dataFp.close()
			print("c1 is", len(c1))
			print("c2 is", len(c2))


			df = pd.DataFrame(zip(startTime, endTime, c1, c2), columns =['start', 'end', "c1", "c2"])
			# Edit our columns to get our desired output DF
			
			return df

def find_time_labels(dataframe, filepath):
	#Check that our filepath is a .txt")
	time, state = [], []
	
	data_an = open(filepath) # open and read file
	r = data_an.read()
	data_an.close()

	rows = list(r.split("\n")) # list of each row (strings)
	index = 0
	for row in rows:
		index += 1
		row_list = list(row.split(","))
		if row_list == [""]:
				break

		if index % 2 == 1: # only want end times
			time.append(np.float32(row_list[2]))
			state.append(int(row_list[5]))

	return [time, state]

def label_dataframe(dataframe, cols, s = 0, e = 10000):
	states_col = []
	time_state_i = 0

	# (0,1), (0,2)
	for index, row in dataframe.iterrows():
		period_start = float(row["start"]) # 1 
		period_end = float(row["end"]) # 2
		time = 0
		state = 1


		# print(f'my index is {index}, current epoch is ({epoch_start},{epoch_end})')
		# if we passed the end time
		# if 2 > 1703.07
		if period_start > float(cols[time][time_state_i]):
				time_state_i += 1
				print(f"time we're on: {cols[0][time_state_i]}")
				print(f'my time is {time_state_i}, current epoch is ({period_start},{period_end})')
		if period_end >= float(cols[time][time_state_i]):
			# 1 < 0.04
			if period_start < float(cols[0][time_state_i]):
				states_col.append(4) # works with artifact : 4

			else:
				states_col.append(cols[1][time_state_i])
		else:
			states_col.append(cols[1][time_state_i])

	return(states_col)

# trying to redo this
# dataframe is [start time, end time, [c1 datapoints], [c2 datapoints]]
# cols is array of arrays of [end time, label]
def label_dataframe_new(dataframe, cols):


	cols_len = len(cols[0])
	assert cols_len == len(cols[1])
 
	labeled_periods = []
	periods_to_label = dataframe.shape[0]
	print("periods to label", periods_to_label)

	last_annotated_label = None
	annotated_index = 0
	# (0,1), (0,2)

	#iterate over the periods
	for period_index, row in dataframe.iterrows():
		
  
		#Case for if we're out of annotations
		if annotated_index >= cols_len:
			if last_annotated_label != None:
					labeled_periods.append(last_annotated_label)
			else:
				# Get a ref to our last label
				last_annotated_label = cols[1][annotated_index -1]
			break
  
		#Annotated end marker
		annotated_end = cols[0][annotated_index]

		# Previous annotated end (set to 0 at the start)
		annotated_start = cols[0][annotated_index-1] if (annotated_index-1 > 0) else 0
		
		# Label of the annotated range
		annotated_label = cols[1][annotated_index]
		

		period_start = row["start"] # 1 
		period_end = row["end"] # 2

		# ==== cases that period and annotated overlap ==== #		
  
		# Period is entirely within annotation range
		if((period_start >= annotated_start) and (period_end <= annotated_end)): # 3
			# print("starts", period_start, ">=", annotated_start, " ends:", period_end, "<=", annotated_end, "label:", cols[1][annotated_index])
			# we advance periods, classify as annotated label
			labeled_periods.append(annotated_label)

		# If period is straddling the left side of the annotation
		elif (period_start < annotated_start and (period_end < annotated_end)): # 4
			# we advance periods append artifact label
			labeled_periods.append(4)
		
		# If period is larger than the entire annotated range
		elif ((period_start <= annotated_start) and (period_end >= annotated_end)): # 2
			# we advance periods append artifact label
			labeled_periods.append(4)

			# Because the period ends after the annotated range, we advance annotated index
			annotated_index += 1

		# If period is straddling the right side of the annotated range
		elif ((period_start > annotated_start) and (period_end > annotated_end)): # 1
      		# we advance periods append artifact label
			labeled_periods.append(4)
	
			# Because the period ends after the annotated range, we advance annotated index
			annotated_index += 1

		
		# ==== cases that period and annotated are disjoint ==== #		
		
		elif (period_start >= annotated_end): # If period is completely after annotated range	
   
			check = annotated_end

			# continue increasing annotated range until annotated has "caught up"
			while (period_start >= check):
				annotated_index += 1
	
				# making the new annotated end to check
				check = cols[0][annotated_index]
			#print("We are here")
			

		# If annotated range is completely after period
		elif(period_end <= annotated_start):
			# This should not happen EVER
			# Artifacting may not always be the correct answer, but it's likely correct
      		# we advance periods append artifact label
			labeled_periods.append(4)			

		else:
			print("some other case! period:", period_start,"-" ,period_end,"annotated:", annotated_start, "-",annotated_end  )

	return labeled_periods