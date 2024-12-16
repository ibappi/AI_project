import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

generation_data = pd.read_csv('./data/Plant_2_Generation_Data.csv')
weather_data = pd.read_csv('./data/Plant_2_Weather_Sensor_Data.csv')
# solar_data = pd.read_csv('./data/solar_datasetv01.csv')
df_solar = pd.read_csv('./data/k_solar_dataset.csv')


# Adjust the format to include seconds (%S)
# generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
# weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
#
# print('Plant generation data', generation_data)
# print('Weather data', weather_data)
#
# # Merging the data
# df_solar = pd.merge(generation_data.drop(columns = ['PLANT_ID']), weather_data.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
# df_solar_merge = df_solar.sample(5).select_dtypes(include=['float64', 'int64'])
#
# plt.figure(figsize=(12, 12))
# sns.heatmap(df_solar_merge, annot=True, cmap='cool', linewidths=0.5)
# plt.show()

# adding separate time and date columns
# df_solar["DATE"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.date
# df_solar["TIME"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.time
# df_solar['DAY'] = pd.to_datetime(df_solar['DATE_TIME']).dt.day
# df_solar['MONTH'] = pd.to_datetime(df_solar['DATE_TIME']).dt.month
# df_solar['WEEK'] = pd.to_datetime(df_solar['DATE_TIME']).dt.isocalendar().week
#
# # add hours and minutes for ml models
# df_solar['HOURS'] = pd.to_datetime(df_solar['TIME'],format='%H:%M:%S').dt.hour
# df_solar['MINUTES'] = pd.to_datetime(df_solar['TIME'],format='%H:%M:%S').dt.minute
# df_solar['TOTAL MINUTES PASS'] = df_solar['MINUTES'] + df_solar['HOURS']*60
#
# # add date as string column
# df_solar["DATE_STRING"] = df_solar["DATE"].astype(str) # add column with date as string
# df_solar["HOURS"] = df_solar["HOURS"].astype(str)
# df_solar["TIME"] = df_solar["TIME"].astype(str)

print(df_solar.head(2))
print(df_solar.info())
print(df_solar.isnull().sum())

# df_solar_numeric = df_solar.select_dtypes(include=['float64', 'int64'])
# print(df_solar_numeric)
# #
# # styled_table = df_solar_numeric.describe().style.background_gradient(cmap='rainbow')
# # styled_table.to_html('styled_table.html')
#
# df_solar_numeric = df_solar.select_dtypes(include=['float64', 'int64'])
# corr = df_solar_numeric.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.show()
# AC/DC/IRRADIATION
# daily_irradiation = df_solar.groupby('DATE')['IRRADIATION'].agg('sum')
#
# daily_irradiation.sort_values(ascending=False).plot.bar(figsize=(17,5), legend=True,color='blue')
# plt.title('IRRADIATION')
# plt.show()

# daily yeild by daily Irridiation
# daily_irradiation = df_solar.groupby('DATE')['IRRADIATION'].agg('sum')
#
# # Compute daily yield (assuming 'DC_POWER' is the column representing power generation)
# daily_yield = df_solar.groupby('DATE')['DC_POWER'].agg('sum')
# daily_data = pd.DataFrame({'Irradiation': daily_irradiation.values, 'Yield': daily_yield.values})
# # Plot both metrics together as bar plots for comparison
# plt.figure(figsize=(10, 6))
#
# # Create a scatter plot
# sns.scatterplot(data=daily_data, x='Irradiation', y='Yield', alpha=0.6)
#
# # Fit a line to the scatter plot to show the trend
# sns.regplot(data=daily_data, x='Irradiation', y='Yield', scatter=False, color='red', ci=None)
#
# plt.title('Relationship Between Daily Irradiation and Daily Yield')
# plt.xlabel('Daily Irradiation (kWh/m²)')
# plt.ylabel('Daily Yield (kWh)')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Daily ambient temperature

# daily_ambient_temp = df_solar.groupby('DATE')['AMBIENT_TEMPERATURE'].agg('sum')
#
# daily_ambient_temp.sort_values(ascending=False).plot.bar(figsize=(17,5), legend=True,color='darkgreen')
# plt.title('AMBIENT_TEMPERATURE')
# plt.show()

# Finding outliers

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df_solar)
# plt.title('Boxplot of Numerical Features')
# plt.xticks(rotation=45)
# plt.show()
# Assuming df_solar is your DataFrame


# def calculate_z_scores(df, threshold=3):
#     z_scores = (df - df.mean()) / df.std()
#     return (z_scores.abs() > threshold).sum()  # Count of outliers
#
#
# # Calculate outliers for specific columns
# outlier_counts = {
#     'DC_POWER': calculate_z_scores(df_solar['DC_POWER']),
#     'AC_POWER': calculate_z_scores(df_solar['AC_POWER']),
#     'DAILY_YIELD': calculate_z_scores(df_solar['DAILY_YIELD']),
#     'TOTAL_YIELD': calculate_z_scores(df_solar['TOTAL_YIELD']),
#     'AMBIENT_TEMPERATURE': calculate_z_scores(df_solar['AMBIENT_TEMPERATURE']),
#     'MODULE_TEMPERATURE': calculate_z_scores(df_solar['MODULE_TEMPERATURE']),
#     'IRRADIATION': calculate_z_scores(df_solar['IRRADIATION'])
# }
#
# print("Total outliers based on Z-score:")
# print(outlier_counts)
#
#
# def remove_outliers_z_score(df, threshold=3):
#     # Create a mask for non-outlier rows
#     mask = np.all(np.abs((df - df.mean()) / df.std()) <= threshold, axis=1)
#     # Filter the DataFrame to only keep non-outlier rows
#     df_no_outliers = df[mask]
#     return df_no_outliers
#
#
# # Remove outliers from specific columns
# columns_to_check = ['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD',
#                     'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
# df_solar_no_outliers = remove_outliers_z_score(df_solar[columns_to_check])
#
# # If you want to keep other columns in the original DataFrame, merge with non-outlier rows
# df_solar_cleaned = df_solar[df_solar.index.isin(df_solar_no_outliers.index)]
#
# # Print the shape of the original and cleaned DataFrames
# print(f"Original DataFrame shape: {df_solar.shape}")
# print(f"Cleaned DataFrame shape: {df_solar_cleaned.shape}")

# z_scores = np.abs(stats.zscore(df_solar_cleaned.select_dtypes(include=['float64', 'int64'])))
#
# # Define a threshold for identifying outliers
# threshold = 3
# outliers_z = (z_scores > threshold)
#
# # Total count of outliers based on Z-score
# total_outliers_z = np.sum(outliers_z)
# print(f"Total outliers based on Z-score: {total_outliers_z}")
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=z_scores)
# plt.title('Boxplot of Numerical Features')
# plt.xticks(rotation=45)
# plt.show()

# columns_to_save = ['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE',
#                    'MODULE_TEMPERATURE', 'IRRADIATION', 'DC_POWER', 'AC_POWER']
#
# final_dataset = df_solar_cleaned[columns_to_save]
#
# # Save to a CSV file
# final_dataset.to_csv('final_dataset.csv', index=False)
#
# print("Final dataset saved as 'final_dataset.csv'")

# print(df_solar.head(2))
# print(df_solar.info())
# print(df_solar.isnull().sum())