import re
import csv

# Read the WhatsApp chat file
with open("dat.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Define a pattern to extract date, time, sender, and message
pattern = r"\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}:\d{2} ?[APM]*)\] (.+?): (.+)"

# Store cleaned data
data = []

for line in lines:
    match = re.match(pattern, line)
    if match:
        date, time, sender, message = match.groups()
        data.append([date, time, sender, message])

# Write to a CSV file
with open("clean_chat.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Date", "Time", "Sender", "Message"])  # Headers
    writer.writerows(data)

print("Data cleaned and saved as clean_chat.csv")
