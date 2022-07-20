import os

folder = "J:\Final"


my_fold = os.listdir(folder)
filtered_path = []

for i in my_fold:
    if "Scene" in i:

        loc = "J:\Final" + "\\" + i
        filtered_path.append(loc)

print(filtered_path)

for i in filtered_path:

    print((i.split("_")[0]).split("\\")[2])
