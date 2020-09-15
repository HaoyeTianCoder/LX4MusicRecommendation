import shutil
import json
import os

root = 'D:/用户目录/下载/MillionSongSubset/data'
todir = 'D:/用户目录/下载/MillionSongSubset/data/all'
track_info1 = 'D:/kaggle/track_info1.json'
track_info2 = 'D:/kaggle/track_info2.json'
track_info3 = 'D:/kaggle/track_info3.json'

#aggregate file
def extract_file():
    for dirpath, dirname, filenames in os.walk(root):
        for filename in filenames:
            file_path = os.path.join(dirpath,filename)
            toname = os.path.join(todir,filename)
            print(file_path,toname)
            shutil.copy(file_path,toname)

def aggregate():
    track_info_dict = {}
    namelist = os.listdir(todir)
    count = 0
    for name in namelist:
        file_path = os.path.join(todir,name)
        with open(file_path,'r',encoding='utf-8') as f:
            dict = json.load(f)
            track = dict['track_id']
            track_info_dict[track] = dict
        count += 1
        if count%10000 == 0:
            print("current count is {:d}".format(count))
        if count == 350000:
            print("writing json1......")
            with open(track_info1, 'w', encoding='utf-8') as f1:
                json.dump(track_info_dict, f1)
            track_info_dict = {}
        if count == 700000:
            print("writing json2......")
            with open(track_info2, 'w', encoding='utf-8') as f2:
                json.dump(track_info_dict, f2)
            track_info_dict = {}

    print("writing json3......")
    with open(track_info3, 'w', encoding='utf-8') as f3:
        json.dump(track_info_dict, f3)

extract_file()
#aggregate()
