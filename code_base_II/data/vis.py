import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DonorsChoose dataset
data = pd.read_csv('test.csv')

# Convert date columns to datetime
date_columns = ['Donation Received Date', 'Project Posted Date', 'Project Fully Funded Date']
for column in date_columns:
    data[column] = pd.to_datetime(data[column], format='%Y-%m-%d %H:%M:%S')

# Time Series Plot: Donation Trends Over Time
data['Donation Received YearMonth'] = data['Donation Received Date'].dt.to_period('M').astype(str)
donation_counts = data['Donation Received YearMonth'].value_counts().sort_index()
# plt.plot(donation_counts.index, donation_counts.values)
# plt.xlabel('Year-Month')
# plt.ylabel('Donation Count')
# plt.title('Donation Trends Over Time')
# plt.xticks(rotation=45)
# plt.show()

# # Histogram of Project Posted Dates
# plt.hist(data['Project Posted Date'], bins='auto')
# plt.xlabel('Project Posted Date')
# plt.ylabel('Frequency')
# plt.title('Distribution of Project Posted Dates')
# plt.xticks(rotation=45)
# plt.show()

# # Seasonal Plot: Average Donation Amount per Quarter
data['Donation Received Quarter'] = data['Donation Received Date'].dt.to_period('Q').astype("str")
average_donation_by_quarter = data.groupby('Donation Received Quarter')['Donation Amount'].mean()
# plt.plot(average_donation_by_quarter.index, average_donation_by_quarter.values)
# plt.xlabel('Quarter')
# plt.ylabel('Average Donation Amount')
# plt.title('Average Donation Amount per Quarter')
# plt.xticks(rotation=45)
# plt.show()

# # Calendar Heatmap: Count of Projects Posted per Date
# project_posted_counts = data['Project Posted Date'].value_counts().reset_index()
# project_posted_counts.columns = ['Project Posted Date', 'Count']
# calendar_heatmap_data = project_posted_counts.set_index('Project Posted Date')
# calendar_heatmap_data['Year'] = calendar_heatmap_data.index.year
# calendar_heatmap_data['Month'] = calendar_heatmap_data.index.month
# calendar_heatmap = calendar_heatmap_data.pivot('Month', 'Year', 'Count')
# sns.heatmap(calendar_heatmap, cmap='Blues')
# plt.xlabel('Year')
# plt.ylabel('Month')
# plt.title('Count of Projects Posted per Date')
# plt.show()

# # Event Timeline: Project Posted and Fully Funded Dates
# timeline_data = data[['Project Posted Date', 'Project Fully Funded Date']].dropna()
# timeline_data['Project Posted'] = 1
# timeline_data['Project Fully Funded'] = 1
# timeline_data = timeline_data.set_index('Project Posted Date').resample('D').sum()
# timeline_data.plot()
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.title('Project Posted and Fully Funded Timeline')
# plt.show()

# Lag Plot: Donation Amount
# pd.plotting.lag_plot(data['Donation Amount'])
# plt.xlabel('Donation Amount (t)')
# plt.ylabel('Donation Amount (t+1)')
# plt.title('Lag Plot of Donation Amount')
# plt.show()

# # Rolling Statistics Plot: Rolling Average Donation Amount
data['Donation Received Date'] = pd.to_datetime(data['Donation Received Date'])
data = data.sort_values(by='Donation Received Date')
rolling_average_donation = data['Donation Amount'].rolling(window=30).mean()
plt.plot(data['Donation Received Date'], rolling_average_donation)
plt.xlabel('Date')
plt.ylabel('Rolling Average Donation Amount')
plt.title('Rolling Average Donation Amount')
plt.xticks(rotation=45)
plt.show()
