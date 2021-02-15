import os


def cleanChineseData(src, dst):
    if os.path.exists(src):
        dirs = os.listdir(src)
        for folder in dirs:
            dirname = src + folder + "/"
            names = os.listdir(dirname)
            with open(dst, 'a', encoding='utf-8') as output:
                for (idx, name) in enumerate(names):
                    if idx > 5000:
                        break
                    fname = dirname + name
                    print(fname)
                    with open(fname, 'r', encoding='utf-8') as f:
                        sentence_list = f.readlines()
                        if '\n' in sentence_list:
                            sentence_list.remove('\n')
                        sentence_list = map(str.strip, sentence_list)
                        for sentence in sentence_list:
                            output.write('%s\n' % sentence)


if __name__ == '__main__':
    cleanChineseData('../../dataset/THUCNews/', "../data/chinese.txt")
