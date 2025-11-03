# importing modules and packages
import pandas as pd
import random
from pathlib import Path

# Constants for seasonal adjustment factors (index 1..12)
MONTH_FACTOR_WEST = [0, 100, 100, 110, 80, 70, 80, 110, 110, 120, 120, 150, 200]
MONTH_FACTOR_EAST = [0, 100, 100, 100, 100, 80, 100, 120, 200, 200, 150, 130, 120]

# Read CSV relative to this script, matching actual filename casing
csv_path = Path(__file__).with_name('nedbor.csv')
df = pd.read_csv(csv_path)

df2 = pd.DataFrame(columns=['X','Y','Nedbor','Month'])
rows = len(df)
for irow in range(0, rows):
        xpos = df.iloc[irow, 0]
        if float(xpos) == 0:
            continue
        ypos = df.iloc[irow, 1]
        for month in range(1, 13):
            # Use EAST factors for x>9, otherwise WEST factors
            factor = MONTH_FACTOR_EAST[month] if xpos > 9 else MONTH_FACTOR_WEST[month]
            nedbor = int(df.iloc[irow, 2] * factor / random.randrange(1220, 2000))
            nedbor = min(499,nedbor)
            df2.loc[len(df2)] = [xpos,ypos,nedbor,month]

df2.to_csv('NedborX.csv',index=False)
print(df2)