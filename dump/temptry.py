import pandas as pd

# Create a Series with default index
s1 = pd.Series([1, 2, 3, 4, 5],[1,2])

# Create a Series with custom index
s2 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

# Print the Series
print(s1)
print(s2)

