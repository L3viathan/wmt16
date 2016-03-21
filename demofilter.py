import sys
import fileinput

for index, line in enumerate(fileinput.input()):
    if "it" in line:
        print(line.strip())
print("done here", file=sys.stderr)
