with open('output.txt', encoding = 'utf-8') as f:
    with open('last-output.txt', 'w', encoding = 'utf-8') as out: 
        for line in f:
            if line.startswith('[') and line.endswith(']\n'):
                out.write(line)
