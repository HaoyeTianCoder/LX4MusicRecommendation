#get data from file

file_path = 'D:/kaggle/kaggle_visible_evaluation_triplets.txt'
new_file = 'D:/kaggle/kaggle_visible_evaluation_triplets_new.txt'
train_nums = 1450933

def dis_listen_times(file_path):
    #view distribution of listen_time
    with open(file_path,'r') as f:
        dict = {}
        for line in f:
            listen_time = line.strip().split('\t')[2]
            if listen_time not in dict.keys():
                dict[listen_time] = 1.0/train_nums
            else:
                dict[listen_time] += 1.0/train_nums
        print (sorted(dict.items(),key=lambda k: k[1],reverse=True))

def fix_label():
    # fix original label
    with open(file_path, 'r',encoding='utf-8') as f1, open(new_file,'w',encoding='utf-8') as f2:
        for line in f1:
            listen_time = int(line.strip().split('\t')[2])
            if listen_time <= 1:
                new_line = line.strip().split('\t')[0] + '\t' + line.strip().split('\t')[1] + '\t' + '0\n'
            else:
                new_line = line.strip().split('\t')[0] + '\t' + line.strip().split('\t')[1] + '\t' + '1\n'
            f2.write(new_line)

def split_data():
    pass

if __name__ == '__main__':
    #fix_label()
    pass
