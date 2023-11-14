
with open("data/shakespeare.txt", "r", encoding="utf-8") as txt_file:
    lines = txt_file.readlines()
with open("data/shakespeare_cleaned.txt", "w", encoding="utf-8") as clean_file:
    for line in lines:
        clean_line = "".join([i for i in line if not i.isdigit()]).strip()
        clean_file.write(f"\n{clean_line}")

