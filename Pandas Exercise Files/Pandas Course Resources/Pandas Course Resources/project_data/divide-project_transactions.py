import pandas as pd

# Read the large CSV
df = pd.read_csv('AIML-Algorithms-Training/Pandas Exercise Files/Pandas Course Resources/Pandas Course Resources/project_data/project_transactions.csv')

# Split into two halves
midpoint = len(df) // 2
df1 = df.iloc[:midpoint]
df2 = df.iloc[midpoint:]

# Save the two parts
df1.to_csv('AIML-Algorithms-Training/Pandas Exercise Files/Pandas Course Resources/Pandas Course Resources/project_data/project_transactions_part1.csv', index=False)
df2.to_csv('AIML-Algorithms-Training/Pandas Exercise Files/Pandas Course Resources/Pandas Course Resources/project_data/project_transactions_part2.csv', index=False)

print(f"Original: {len(df)} rows")
print(f"Part 1: {len(df1)} rows")
print(f"Part 2: {len(df2)} rows")