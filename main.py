import os

print("Hello World")

# GENERIC PLACEHOLDER UNTIL OUR MODEL TYPE IS DECIDED
# TODO: UPDATE W/ APPROPRIATE NAME
model_name = "NeuralNets"
name = f"model/NeuralNets/{model_name}"
log_dir = f"{name}"

data_dir = "data/train.txt"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# READ DATA FROM FILE INTO INPUT & LABELS
# TODO: DECIDE HOW WE WILL PROCESS NEW LINES
file = open(data_dir, "r")
data = []
labels = []
for line in file:
    word = line.split(' ')
    
    if word[0] == '\n':
        # print("NEW LINE")
        continue
    else:
        data.append(word[0])
        labels.append(word[1])

# print(len(data) == len(labels))