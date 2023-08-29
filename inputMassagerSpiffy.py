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



		

# Takes: data filepath, periodSize, maximumPeriods(optional)
# Returns: dataframe with columns ['start', 'end', "c1", "c2"]
# Makes periods of the same size from the text file
def makePeriodFromTxt(filepath, periodSize, maxPeriods=None):

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
			# print("c1 is", len(c1))
			# print("c2 is", len(c2))


			df = pd.DataFrame(zip(startTime, endTime, c1, c2), columns =['start', 'end', "c1", "c2"])
			# Edit our columns to get our desired output DF
			
			return df

# Takes:    annotated data filepath
# Returns:  list of the format [<annotated_timestamp>,<annotated_state>]
# We ignore the start timestamps, and just manage stop timestamps
def find_time_labels(filepath):
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



# Takes:   dataframe, list
# Returns: 3 Tensors: labels, c1, c2 
# dataframe with columns ['start', 'end', "c1", "c2"], list of the format [<annotated_timestamp>,<annotated_state>]
# cols is array of arrays of [end time, label]
# percentage is the fraction the largest classification in an artifacted period must be for the whole period to be classified as such
# otherwise it is classified as an artifact
def label_dataframe_new(dataframe, cols, period_size, percentage=1.0):

	cols_len = len(cols[0])

	#Sanity check, c1 and c2 should be the same length
	assert cols_len == len(cols[1])

	labeled_periods = []
	periods_to_label = dataframe.shape[0]
	last_annotated_label = None
	annotated_index = 0
 
	#Annotated end marker
	annotated_end = cols[0][annotated_index]

    # Previous annotated end (set to 0 at the start)
	annotated_start = cols[0][annotated_index-1] if (annotated_index-1 > 0) else 0
   
    # Label of the annotated range
	annotated_label = cols[1][annotated_index]

    #iterate over the periods
	for period_index, row in dataframe.iterrows():
        #Case for if we're out of annotations
		if (annotated_index >= cols_len):
			if (last_annotated_label != None):
                    # If we have a reference to our last annotated label, we assume the rest is that label
                    # Just append last annotated label until we run out of periods
					labeled_periods.append(last_annotated_label)
			else:
                # Get a ref to our last annotated label
				last_annotated_label = cols[1][annotated_index -1]
				labeled_periods.append(last_annotated_label)
			continue
   	
		period_start = row["start"] # 1
		period_end = row["end"] # 2
       
        # Period is entirely within annotation range
		if ((period_start >= annotated_start) and (period_end <= annotated_end)): # 3
            # print("starts", period_start, ">=", annotated_start, " ends:", period_end, "<=", annotated_end, "label:", cols[1][annotated_index])
            # we advance periods, classify as annotated label
			labeled_periods.append(annotated_label)
		#artifact analysis
		else:
			#we can assume that annotated start is larger or equal to period_start
			cTimes = {}
			while (annotated_end < period_end):
				if (period_start >= annotated_start):
					time = annotated_end - period_start
				else:
					time = annotated_end - annotated_start
				cTimes.update({annotated_label:(time + cTimes.get(annotated_label, 0))})
                #fix when you increment in case of overlap
				if (annotated_end <= period_end):
					annotated_index += 1
					if (annotated_index < len(cols[0])):
                        #Annotated end marker
						annotated_end = cols[0][annotated_index]
                        # Previous annotated end (set to 0 at the start)
						annotated_start = cols[0][annotated_index-1] if (annotated_index-1 > 0) else 0
                        # Label of the annotated range
						annotated_label = cols[1][annotated_index]
					else:
						#no more annotated data
						annotated_start = period_start
						annotated_end = period_end
			#why doesn't it go to the end of the annotated data?
			print(period_end - period_start)
			print(cTimes, period_end, period_start, annotated_start, annotated_end)
			time = period_end - annotated_start
			cTimes.update({annotated_label:(time + cTimes.get(annotated_label, 0))})
			total = 0
			for t in cTimes.values():
				total += t
			if ((cTimes.get(max(cTimes, key=cTimes.get)) / total) >= percentage):
				label = max(cTimes, key=cTimes.get)
			else:
				label = 0
			labeled_periods.append(label)			

		# debugging else case, not needed
		# else:
		# 	print("some other case! period:", period_start,"-" ,period_end,"annotated:", annotated_start, "-",annotated_end  )
		#make appropriately shaped tensors for the eeg and emg data
	#initialize empty tensors of size [num of samples, 1, period_size]
	eeg_tensor = torch.zeros((len(dataframe.c1.values), 1, period_size)) 
	emg_tensor = torch.zeros((len(dataframe.c2.values), 1, period_size))

	#add each sample of data to the eeg and emg tensors
	for i in range(len(dataframe.c1.values)):
		eeg_tensor[i, 0] = torch.tensor(dataframe.c1.values[i])
		emg_tensor[i, 0] = torch.tensor(dataframe.c1.values[i])	
	
	return torch.tensor(labeled_periods), eeg_tensor, emg_tensor

def get_fourier_transform(dataframe, col):
	fft_tensor = torch.zeros((len(dataframe[col].values), 1, 200))

	for index, row in dataframe.iterrows():
    	#pt = torch.tensor(dataframe[col])
		fft_tensor[index, 0] = torch.fft.fft(torch.tensor(row[col]))

	return fft_tensor
    


# Takes:   data filepath, annotated data filepath, and period size
# Returns: 3 Tensors: labels, c1, c2 
def get_labeled_data(data_filepath, annotated_filepath, period_size,  maxPeriods=None):


	# Get intermediate df with c1, c2, and times
	intermediateDf = makePeriodFromTxt(data_filepath, period_size)

	# Get annotated time labels
	annotated_labels = find_time_labels(annotated_filepath)

	# Label our created periods and get our output tensors
	labels, c1, c2 = label_dataframe_new(intermediateDf, annotated_labels, period_size)
	
	#generate foureir transformed data for eeg signal, same format/dimesions as c1 and c2
	eeg_FFT = get_fourier_transform(intermediateDf, 'c1')

    	#generate foureir transformed data for emg, same format/dimesions as c1 and c2
	emg_FFT = get_fourier_transform(intermediateDf, 'c2')

	return labels, c1, c2, eeg_FFT, emg_FFT
