import os

# Specify the directory path
directory_path = "drive/MyDrive/LI-Jobs-JSON/"

# Get a list of all files in the directory
files = os.listdir(directory_path)

# Iterate over each file and rename it
for old_name in files:
    if old_name[0] == 'o':
      # Construct the new name however you like
      new_name = "r" + old_name  # Example: Add a prefix to the old name

      # Join the directory path with the old file name
      old_path = os.path.join(directory_path, old_name)

      # Join the directory path with the new file name
      new_path = os.path.join(directory_path, new_name)

      # Rename the file
      os.rename(old_path, new_path)

      #print(f"Renamed '{old_name}' to '{new_name}'")

print("All files have been renamed.")