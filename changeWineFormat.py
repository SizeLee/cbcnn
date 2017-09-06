import re

if __name__ == '__main__':
    readfp = open('..\\balance-scale.rawdata.txt', 'r')
    writefp = open('balance-scale.txt', 'w')

    regex = re.compile(r'^\.[0-9]+$')
    for line in readfp:
        words = line.strip().split(',')
        length = len(words)
        # print(length)
        wtl = ''

        for i in range(length):

            if regex.match(words[(i+1) % length]):
                words[(i + 1) % length] = '0' + words[(i+1) % length]

            # print(i + 1)
            if i+1 == length:
                words[(i + 1) % length] = 'class' + words[(i + 1) % length]

            wtl = wtl + words[(i + 1) % length]
            # print(wtl)


            if i+1 != length:
                wtl = wtl + ','

            else:
                wtl = wtl + '\n'

        if words[0] == 'classB':
                wtl = wtl * 6

        writefp.write(wtl)

    readfp.close()
    writefp.close()
