# Importing the pandas library and aliasing it as 'pd'
import pandas as pd

# Reading a CSV file named 'book_sales.csv' into a pandas DataFrame
# Setting the 'Date' column as the index and parsing it as dates
# Dropping the 'Paperback' column from the DataFrame using the drop method
df = pd.read_csv(
    "book_sales.csv",
    index_col='Date',  # Setting 'Date' column as the index
    parse_dates=['Date'],  # Parsing 'Date' column as dates
).drop('Paperback', axis=1)  # Dropping the 'Paperback' column along the specified axis

# Displaying the first few rows of the DataFrame using the 'head()' method
df.head()

"""This series records the number of hardcover book sales at a retail store over 30 days.



---

Linear regression is widely used in practice and adapts naturally to even complex forecasting tasks.

The linear regression algorithm learns how to make a weighted sum from its input features. For two features, we would have:

```
target = weight_1 * feature_1 + weight_2 * feature_2 + bias
```

The regression algorithm learns values for the parameters weight_1, weight_2, and bias that best fit the target. (This algorithm is often called ordinary least squares since it chooses values that minimize the squared error between the target and the predictions.) The weights are also called regression coefficients and the bias is also called the intercept because it tells you where the graph of this function crosses the y-axis.

#Time-step features
There are two kinds of features unique to time series: time-step features and lag features.

Time-step features are features we can derive directly from the time index. The most basic time-step feature is the time dummy, which counts off time steps in the series from beginning to end.
"""

# Importing the NumPy library with the alias 'np'
import numpy as np

# this code adds a new column 'Time' to the DataFrame
# The values in the 'Time' column are generated using NumPy's 'arange' function
# 'len(df.index)' returns the number of rows in the DataFrame, and 'arange' creates an array of that length
# This array represents a sequence of numbers starting from 0 up to (length-1)
df['Time'] = np.arange(len(df.index))

# Displaying the first few rows of the DataFrame using the 'head()' method
df.head()

"""Linear regression with the time dummy produces the model:

`target = weight * time + bias`

The time dummy then lets us fit curves to time series in a time plot, where Time forms the x-axis.
"""

# Commented out IPython magic to ensure Python compatibility.
# Import necessary libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plots using seaborn and configure some default settings for matplotlib
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

# Set the figure format for inline plotting
# %config InlineBackend.figure_format = 'retina'

# Create a subplot using matplotlib
fig, ax = plt.subplots()

# Plot a line chart of 'Hardcover' sales over 'Time' using a light gray color (0.75)
ax.plot('Time', 'Hardcover', data=df, color='0.75')

# Overlay a regression plot (scatter plot with a fitted line) on the same axes using seaborn
# Set the x-axis to 'Time', y-axis to 'Hardcover', disable confidence intervals (ci=None),
# and customize scatter plot appearance with a dark gray color (0.25)
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))

# Set the title of the plot
ax.set_title('Time Plot of Hardcover Sales');

"""#Lag features
To make a lag feature we shift the observations of the target series so that they appear to have occured later in time. Here we've created a 1-step lag feature, though shifting by multiple steps is possible too.
"""

# this line creates a new column 'Lag_1'
# in the DataFrame 'df' and assigns it the values of the 'Hardcover' column shifted by 1 position.
df['Lag_1'] = df['Hardcover'].shift(1)

# Reindexes the columns of the DataFrame to include only 'Hardcover' and 'Lag_1'.
# This reordering doesn't affect the data but helps in organizing the DataFrame.
df = df.reindex(columns=['Hardcover', 'Lag_1'])

# Displays the first few rows of the DataFrame after the above operations.
# This helps in inspecting the changes made to the DataFrame.
df.head()

"""Linear regression with a lag feature produces the model:

`target = weight * lag + bias`


So lag features let us fit curves to lag plots where each observation in a series is plotted against the previous observation.
"""

# Importing necessary libraries
import matplotlib.pyplot as plt  # Matplotlib for basic plotting
import seaborn as sns            # Seaborn for statistical data visualization

# Creating a scatter plot with a regression line using seaborn
# 'Lag_1' is plotted on the x-axis, 'Hardcover' on the y-axis using data from the DataFrame 'df'
# 'ci=None' specifies that no confidence intervals should be displayed for the regression line
# 'scatter_kws=dict(color='0.25') sets the color of the scatter points to a light gray (RGB value: 0.25)
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))

# Setting the aspect ratio of the plot to 'equal'
# This ensures that the scale of the x-axis is the same as the scale of the y-axis
ax.set_aspect('equal')

# Adding a title to the plot
ax.set_title('Lag Plot of Hardcover Sales')

# Displaying the plot
plt.show()

