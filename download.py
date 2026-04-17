import kagglehub

# Download latest version
path = kagglehub.dataset_download("utsavsinghal2604/eeg-motorimagery-eegmmidb-v1-0-0")

print("Path to dataset files:", path)