import csv

# Define input and output file paths
input_file = "instances.csv"
output_file = "instances2.csv"

# Open the input and output files
with open(input_file, "r") as csv_in, open(output_file, "w", newline="") as csv_out:
    reader = csv.reader(csv_in)
    writer = csv.writer(csv_out)
    
    # Write only non-empty rows to the output file
    for row in reader:
        if any(row):  # Checks if the row is not empty
            writer.writerow(row)