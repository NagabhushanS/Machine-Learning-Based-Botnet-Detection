"""Author: Nagabhushan S Baddi (InStep Intern)\nOrganization: Infosys Ltd.\nGUI app (module) to detect botnet traffic using Machine Learning"""

#import the modules
from Tkinter import *
from ttk import *
from tkFileDialog import *
import dataset_load
import models
import threading
import time
import pickle

#load data set
file = open('../dataset/flowdata.pickle', 'rb')
sd = pickle.load(file)
X, Y, XT, YT = sd[0], sd[1], sd[2], sd[3]

def callSuitable(mlalgo, v2):
    """use and evaluate the selected Machine Learning algorithm"""
    global X, Y, XT, YT

    if mlalgo == 'Decision Tree':
        model = models.DTModel(X, Y, XT, YT, v2)
        model.start()
    elif mlalgo == 'Naive Bayes':
        model = models.NBModel(X, Y, XT, YT, v2)
        model.start()
    elif mlalgo == 'SVM':
        model = models.SVMModel(X, Y, XT, YT, v2)
        model.start()
    elif mlalgo == 'K Nearest Neighbours':
        model = models.KNNModel(X, Y, XT, YT, v2)
        model.start()
    elif mlalgo == 'Logistic Regression':
        model = models.LogModel(X, Y, XT, YT, v2)
        model.start()
    else:
        model = models.ANNModel(X, Y, XT, YT, v2)
        model.start()

if __name__ == "__main__":
    #code for the GUI
    root = Tk()
    root.title('Botnet Detection Using Machine Learning')
    root.resizable(width=False, height=False)

    frame1 = Frame(root, padding=(0, 0, 0, 0), width=300)

    label = Label(frame1, text='Dataset File:    \n(.binetflow file)')
    label.grid(row=0, column=0, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))

    v = StringVar(frame1, value='.binetflow')
    entry = Entry(frame1, textvariable = v)
    entry.grid(row=0, column=1, rowspan=1, columnspan=1, padx=10, pady=10, sticky=(W, E))

    button = Button(frame1, text='Browse')
    button.bind('<1>', lambda e: v.set(askopenfilename().split('/')[-1]))
    button.grid(row=0, column=2, rowspan=1, columnspan=1, padx = 10, pady=10)


    machineLabel = Label(frame1, text='Machine Learning\nAlgorithms')
    machineLabel.grid(row=1, column=0, padx=10, pady=10, sticky=(W, ))

    combo = Combobox(frame1)
    combo['values'] = sorted(['ANN', 'Decision Tree', 'SVM', 'K Nearest Neighbours', 'Naive Bayes', 'Logistic Regression'])
    combo.grid(row=1, column=1, padx=10, pady=10)

    v2 = StringVar(frame1, value='Accuracy: ')
    resultLabel = Label(frame1, textvariable=v2)

    calButton = Button(frame1, text='Go')
    calButton.bind('<1>', lambda e: callSuitable(combo.get(), v2))
    calButton.grid(row=1, column=2, sticky = (E, W), padx=10, pady=10)


    resultLabel.grid(row=2, pady=10, padx=10, columnspan=3, sticky=(W, ))

    frame1.grid()

    #run the GUI
    root.mainloop()


