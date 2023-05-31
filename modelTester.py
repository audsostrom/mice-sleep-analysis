from inputMassager import *
import numpy as np
import tkinter as tk #tk for file dialog (requires Jinja2!!!)
from tkinter import filedialog #tkinter for file dialog
from modelTesterGUI import testerGUI

#initalize GUI
guiObj = testerGUI()

# We can do event binding to pass keypresses to the gui
# testObj.bind('<Configure>', testObj.onSizeChange)
# testObj.bind('<r>', testObj.onResizeEvent)
#testObj.bind('<ButtonRelease-1>', testObj.resizer)

#Call guiObj main loop
guiObj.mainloop()

