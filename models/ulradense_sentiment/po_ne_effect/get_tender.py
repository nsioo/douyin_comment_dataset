# coding = utf-8

import os




def get_tend(path):

    if os.path.exists('myPos.txt'):
        os.remove('myPos.txt')
    if os.path.exists('myNeg.txt'):
        os.remove('myNeg.txt')


    with open(path, 'r', encoding='utf-8') as f:
        for i in f:
            a = i.encode(encoding='utf-8')

            line = str(a).split('\\t')
            if '+' in line[1]:
                for x in line[2].split(','):
                    if not os.path.exists('myPos.txt'):
                        with open('myPos.txt', 'w', newline='\n', encoding='utf-8') as wf:
                            wf.write(x+'\r\n')
                    else:
                        with open('myPos.txt', 'a', newline='\n', encoding='utf-8') as wf:
                            wf.write(x+'\r\n')
            elif '-' in line[1]:
                for x in line[2].split(','):
                    if not os.path.exists('myNeg.txt'):
                        with open('myNeg.txt', 'w', newline='\n', encoding='utf-8') as wf:
                            wf.write(x+'\r\n')
                    else:
                        with open('myNeg.txt', 'a', newline='\n', encoding='utf-8') as wf:
                            wf.write(x+'\r\n')
        print('success')




if __name__ == "__main__":
    path = 'goldStandard.tff'

    get_tend(path)