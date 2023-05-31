import tkinter as tk
from tkinter import filedialog
#from tkinter import ttk
import math # for floor

#placeholder command for unused menu buttons
def donothing():
	print("this does nothing!")



# Class AtlasGUI extends tk.Tk, allowing the use of components present in the Tk() class!
class testerGUI(tk.Tk):
    # Init function allows acceptance of arbitrary number of args to support our parent class
    # not yet sure if this is nessecary
    def __init__(self, *args, **kwargs):
        #Init our parent tk class
        tk.Tk.__init__(self, *args, **kwargs)


        # ===== Window tkinter details ===== #
        self.title("Sleep Mice Analysis")
        self.geometry("1000x600")
        self.config(bg = "white")

        # ===== Handling image variables ===== #

        # This list holds all our data filepaths
        # It holds a tuples of (original_data, annotated_data)
        self.dataFilepaths = []

        # ===== Adding all Menus to the window ===== #
        # MenuBar is our main bar containing subsequent menus
        menuBar = tk.Menu(self)

        # Subsequent menus defined here
        filemenu = tk.Menu(menuBar, tearoff=0)

        # Filemenu commands
        filemenu.add_command(label="Add Annotated & Data", command=self.add_data)
        filemenu.add_command(label="Remove Annotated & Data ", command=self.remove_data)
        filemenu.add_command(label="Other?", command=donothing)
        # Add menuBar to AtlasGUI
        self.config(menu = menuBar)
        # Attach subsequent menus to menubar
        menuBar.add_cascade(label="File", menu=filemenu)

        # === Model selection === #
        # Create a list of machine learning models
        models = ["CNN_A", "CNNN_B", "LSTM"]

        # Listbox to display currently selected filepaths
        self.fileListbox = tk.Listbox(self, bg = "yellow", selectbackground="lightblue")
        self.fileListbox.pack(side="right", fill="both", expand=True)

        # Create tkinter BooleanVar variables to store selection status
        self.model_vars = [tk.BooleanVar() for _ in models]

        # Create Checkbuttons for model selection
        for i, model in enumerate(models):
            checkbox = tk.Checkbutton(self, text=model, variable=self.model_vars[i], bg="grey", width=20, height=2)
            checkbox.pack(anchor="w")


        self.runButton = tk.Button(self, text="Run",  bg="green", width=20, height=5, command=self.runModels)
        self.runButton.config(font=("Arial", 14), fg="white")

        # Pack the button to the bottom
        self.runButton.pack(side="left", anchor="sw")

    # Runs the selected models
    def runModels(self):
        print("selected models:" + str(self.model_vars[0].get()) +
            " " + str(self.model_vars[1].get()) +
            " " + str(self.model_vars[2].get()))

     # Opens file selection window and adds selected filepaths to the list
    def add_data(self):
        data_filepath = filedialog.askopenfilename(title = "Select Datafile")
        annotated_filepath = filedialog.askopenfilename(title = "Select Associated Annotated File")

        # Check that both were given
        if((data_filepath != "") and (annotated_filepath != "")):
            # Add data files to our list
            self.dataFilepaths.append((data_filepath, annotated_filepath))
            # Add data files to our shown listbox (we index at tk.END to get the last index to add to)
            new_string = f"data:{data_filepath}\nannotated:{annotated_filepath}"
            self.fileListbox.insert(tk.END, new_string)

    # Removes selected filepaths from the list and the listbox
    def remove_data(self):
        # .curselection() retrieves all selected indices in the currently selected listbox
        selected_indices = self.fileListbox.curselection()
        for index in reversed(selected_indices):
            # Get the filepath from Listbox and remove it from our list
            filepath = self.fileListbox.get(index)
            self.dataFilepaths.remove(filepath)
            # Delete the listBox element at index
            self.fileListbox.delete(index)