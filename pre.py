with open('output.txt') as f:
    with open('last-output.txt', 'w') as out: 
        for line in f:
            if line.startswith('[') and line.endswith(']\n'):
                out.write(line)
