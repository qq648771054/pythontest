t = [1, 2, 3, 4, 5, 6, 5, 6, 9, 1, 4, 5, 6]
for i, x in enumerate(t):
    if x == 5:
        t.pop(i)
print(t)