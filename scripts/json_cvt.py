import json

res = {"result": []}
words = []
line = []
with open("./tt.bmes", "r", encoding="utf-8") as f:
    for i in f.readlines():
        if i != "\n":
            line.append(i[0])
        else:
            words.append(line)
            line = []

tags = []
with open("./tt.json", "r", encoding="utf-8") as f:
    for i in f.readlines():
        d = json.loads(i)["tag_seq"]
        tags.append(d.split(" "))

assert len(words) == len(tags)
for i in range(len(tags)):
    for j in range(len(words[i])):
        res["result"].append(f"{words[i][j]} {tags[i][j]}\n")
    res["result"].append("\n")

with open("./res.txt", "w", encoding="utf-8") as f:
    f.writelines(res.get("result"))
