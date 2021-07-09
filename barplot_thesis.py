"""
====================
Horizontal bar chart
====================

This example showcases a simple horizontal bar chart.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/Ruben/Documents/Studie/Afstudeerproject/Code/Hoeveel_img_per_style.csv")
print(df)

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = df["Style"]
y_pos = np.arange(len(people))
performance = df[" number of images"]

ax.barh(y_pos, performance, align='center', tick_label=True)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Images')
ax.set_title('Number of images per style')

plt.show()
