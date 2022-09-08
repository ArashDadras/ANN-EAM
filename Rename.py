import os

# Function to rename multiple files in directory
def rename_files(directory): 
    for count, filename in enumerate(os.listdir(directory)):
        dst = f"{str(count + 1)}_DATA.LDT"
        src =f"{directory}/{filename}"
        dst =f"{directory}/{dst}"
        os.rename(src, dst)
        
if __name__ == '__main__':
    rename_files('./behavioral_data/selected_data')