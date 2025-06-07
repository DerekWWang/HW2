import random
import csv

p1 = 97
p2 = 113


# Generates one string in the format "a + b=c mod p"
def generate_add(p):
    c = random.randint(0, p)

    if c == p:
        return f"0 + 0 = {p}"
    
    a = random.randint(0, c)
    b = c - a

    return f"{a} + {b} = {c}"

def generate_sub(p):
    a = random.randint(0, p)
    b = random.randint(0, a)
    c = a - b

    return f"{a} - {b} = {c}"

def generate_div(p):
    def get_multiples(n: int) -> tuple[int, int]:
        res = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                res.add(i)
                res.add(n // i)

        choice = random.choice(list(res))
        choice_d =  n / choice
        assert(choice_d % 1 == 0.0)
        return (choice, int(choice_d)) 

    n = random.randint(1, p)
    b, c = get_multiples(n)

    return f"{n} / {b} = {c}"
    

def generate_batch(d = 1000, filepath=''):
    res = []
    for _ in range(d):
        r = random.randint(1,3)
        p = p1 if random.random() >= 0.5 else p2
        if r == 1:
            res.append(f"<s>{generate_add(p)}<e>")
        elif r == 2:
            res.append(f"<s>{generate_sub(p)}<e>")
        else:
            res.append(f"<s>{generate_div(p)}<e>")

    if filepath != '':
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\n')
            writer.writerow(res)

    return res

def generate_add_sub(d = 1000, p=97, filepath=''):
    res = []
    for _ in range(d):
        r = random.random()
        if r < 0.5:
            res.append(f"<s>{generate_add(p)}<e>")
        else:
            res.append(f"<s>{generate_sub(p)}<e>")
    if filepath != '':
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\n')
            writer.writerow(res)

    return res

# generate_batch(1000, True)