import pandas as pd
import cv2
import numpy as np
import tkinter
from tkinter.ttk import *
from PIL import ImageTk, Image

global df
try:
    df = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/Annotation.csv")
except:
    # nog fixen
    df = pd.DataFrame({"path": "Impressionism/claude-monet_the-fonds-at-varengeville.jpg", "id":29894, "Annotation":1})

og = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/id.csv")
rootmap = "C:/Users/Ruben/Documents/Studie/Afstudeerproject/wikiart/wikiart"

def add_to(type, root, loc_path, cur_id):
    # df.at[index, 'Annotation'] = type
    global df
    df = df.append({'path': loc_path, 'id': cur_id, "Annotation" : type}, ignore_index=True)
    root = root.destroy()

try:
    print("Portraits , Landscapes")
    print(df["Annotation"].value_counts()[0], ", ", df["Annotation"].value_counts()[1])
    
    reccomendations = np.load("annotation_reccomendation.npy")
    # count = 0
    while df["Annotation"].value_counts()[0] < 500 or df["Annotation"].value_counts()[1] < 500 and len(reccomendations) > 0:
        # print(df.tail())
        # random_image = random.choice(range(len(og)))
        random_image = reccomendations[0]
        reccomendations = np.delete(reccomendations, 0)
        if not random_image in df["id"].to_list():
            loc_path = og["path"][random_image]
            cur_id = random_image
            image_path = rootmap + "/" + loc_path
            root = tkinter.Tk()
            root.geometry("1800x1000")
            btn = Button(root, text = 'Portrait',
                                    command =lambda: add_to(0, root, loc_path, cur_id))
            btn.grid(row = 0, column = 1, sticky = "W", pady = 2)
            btn = Button(root, text = 'Landscape',
                                    command =lambda: add_to(1, root, loc_path, cur_id))
            btn.grid(row = 0, column = 1, sticky = "N", pady = 2)
            btn = Button(root, text = 'Other',
                                    command =lambda: add_to(2, root, loc_path, cur_id))
            btn.grid(row = 0, column =1, sticky = "E", pady = 2)
            img = Image.open(image_path)
            width, height = img.size
            img = img.resize((round(800/width*height) , round(800)))
            img = ImageTk.PhotoImage(img)
            panel = tkinter.Label(root, image = img)
            panel.grid(row = 1, column = 1, sticky = "S", pady = 20)
            root.mainloop()
            # count += 1
 
        print(df["Annotation"].value_counts()[0], ", ", df["Annotation"].value_counts()[1])
    df.drop_duplicates(subset=["id"])
    df.to_csv('C:/Users\Ruben\Documents\Studie\Afstudeerproject\Code\Annotation.csv', index=False)
except KeyboardInterrupt:
    df.drop_duplicates(subset=["id"])
    df.to_csv('C:/Users\Ruben\Documents\Studie\Afstudeerproject\Code\Annotation.csv', index=False)
    pass