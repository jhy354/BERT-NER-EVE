out = []
with open("test.char.bmes", "r", encoding="utf-8") as f:
    for i in f.readlines():
        if i != "\n":
            out.append(i[0] + " O")
        else:
            out.append(i)
    
with open("11dev.char.bmes", "w", encoding="utf-8") as f:
    for i in out:
        f.write(i)
        if i != "\n":
            f.write("\n")
            
