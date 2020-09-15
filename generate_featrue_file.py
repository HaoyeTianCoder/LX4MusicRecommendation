import json
from config import *
import h5py

# calcu popularity of song
def gen_song_popularity():
    popularity_dict = {}
    with open(train_v,'r',encoding='utf-8') as f:
        for line in f:
            _, song, _ = line.strip().split('\t')
            if song not in popularity_dict.keys():
                popularity_dict[song] = 1
            else:
                popularity_dict[song] += 1
    with open(song_popularity,'w',encoding='utf-8') as f2:
        json.dump(popularity_dict,f2)

#generate songid:artist
def gen_song_artist():
    artist_dict = {}
    with open(unique_tracks,'r',encoding='utf-8') as f:
        for line in f:
            trackid,songid,artist,title = line.strip().split('<SEP>')
            artist_dict[songid] = artist
    with open(song_artist,'w',encoding='utf-8') as f2:
        json.dump(artist_dict,f2)

#generate user_like, such as {userid:{artist1:like1, artist2:like2, artist3:like3}}
def gen_user_like():
    with open(song_artist,'r',encoding='utf-8') as f:
        song_artist_dict = json.load(f)
    user_like_dict = {}
    #get user:{artist:count}
    with open(original_file,'r',encoding='utf-8') as f2:
        for line in f2:
            user, song, count = line.strip().split('\t')
            artist = song_artist_dict[song]
            if not user in user_like_dict.keys():
                user_like_dict[user] = {artist:float(count)}
            else:
                if not artist in user_like_dict[user].keys():
                    user_like_dict[user][artist] = float(count)
                else:
                    user_like_dict[user][artist] += float(count)
    #get user:{artist:like}
    for user,like in user_like_dict.items():
        top5artist = sorted(like.items(),key=lambda x:x[1],reverse=True)[:5]
        top5artist_dict = {}
        # count = 0.0
        # for kv in top5artist:
        #     if float(kv[1]) > 5:
        #         count += float(kv[1])
        for kv in top5artist:
            if float(kv[1]) > 3:
                top5artist_dict[kv[0]] = float(kv[1])
        user_like_dict[user] = top5artist_dict

    with open(user_like,'w',encoding='utf-8') as f2:
        json.dump(user_like_dict,f2)

#generate song:track
def gen_song_track():
    with open(unique_tracks,'r',encoding='utf-8') as f:
        song_track_dict = {}
        for line in f:
            track,song,_,_ = line.strip().split('<SEP>')
            song_track_dict[song] = track
    with open(song_track,'w',encoding='utf-8') as f2:
        json.dump(song_track_dict,f2)

#generate track:year
def gen_track_year():
    with open(track_year_txt,'r',encoding='utf-8') as f:
        track_year_dict = {}
        for line in f:
            year,track,_,_ = line.strip().split('<SEP>')
            track_year_dict[track] = year
    with open(track_year,'w',encoding='utf-8') as f2:
        json.dump(track_year_dict,f2)

#generate song:year
def gen_song_year():
    song_year_dict = {}
    with open(song_track,'r',encoding='utf-8') as f:
        song_track_dict = json.load(f)
    with open(track_year,'r',encoding='utf-8') as f2:
        track_year_dict = json.load(f2)
    with open(train_v,'r',encoding='utf-8') as f3:
        for line in f3:
            _,song,_ = line.strip().split('\t')
            track = song_track_dict[song]
            year = None
            if track in track_year_dict.keys():
                year = 2018-float(track_year_dict[track])
            song_year_dict[song] = year
    with open(song_year,'w',encoding='utf-8') as f4:
        json.dump(song_year_dict,f4)

def gen_track_info():
    track_list_dict = {}
    with open(song_track,'r',encoding='utf-8') as f:
        song_track_dict = json.load(f)
    f = h5py.File(hdf5_file)
    group_analysis = f['analysis']
    group_metadata = f['metadata']
    group_musicbrainz = f['musicbrainz']

    analy_numpy = group_analysis['songs'].value
    meta_numpy = group_metadata['songs'].value
    musi_numpy = group_musicbrainz['songs'].value

    for index in range(len(analy_numpy)):
        line = analy_numpy[index]
        meta_line = meta_numpy[index]
        trackid = str(line[-1], encoding='utf-8')
        duration = round(float(line[3])/60,0)
        end_of_fade_in = str(int(line[4]))
        loudness = int(line[-8])
        mode = int(line[-7])
        mode_confi = float(line[-6])
        tempo = str(int(line[-4]))
        time_sig = str(line[-3])
        time_sig_confi = float(line[-2])

        artist_hottness = round(float(meta_line[3]),2)
        song_hottness = round(float(meta_line[-4]),2) if meta_line[-4] != None else None

        list1 = [duration,end_of_fade_in,loudness,mode,mode_confi,tempo,time_sig,time_sig_confi,artist_hottness,song_hottness]
        list1 = list(map(float,list1))
        track_list_dict[trackid] = list1
    with open(track_list,'w',encoding='utf-8') as f:
        json.dump(track_list_dict,f)

#generate user:[genre]
def gen_user_like_genre():
    user_like_genre_dict = {}
    with open(song_track,'r',encoding='utf-8') as f:
        song_track_dict = json.load(f)
    with open(original_file,'r',encoding='utf-8') as f2:
        for line in f2:
            user,song,count = line.strip().split('\t')
            track = song_track_dict[song]
            genre_file = genre_dir + '/' + track + '.json'
            try:
                with open(genre_file,'r',encoding='utf-8') as f3:
                    genre_dict = json.load(f3)
            except:
                continue
            tags = genre_dict['tags']
            tag_dict = dict(tags)
            for k,v in tag_dict.items():
                tag_dict[k] = float(v) * float(count)
            if not user in user_like_genre_dict.keys():
                user_like_genre_dict[user] = tag_dict
            else:
                for k,v in tag_dict.items():
                    if not k in user_like_genre_dict[user]:
                        user_like_genre_dict[user][k] = float(v)
                    else:
                        user_like_genre_dict[user][k] += float(v)

    # get user{genre:like}
    for user, like in user_like_genre_dict.items():
        top5artist = sorted(like.items(), key=lambda x: x[1], reverse=True)[:5]
        top5artist_dict = {}
        # count = 0.0
        # for kv in top5artist:
        #     if float(kv[1]) > 5:
        #         count += float(kv[1])
        for kv in top5artist:
            if float(kv[1]) >= 100:
                top5artist_dict[kv[0]] = float(kv[1])
                user_like_genre_dict[user] = top5artist_dict

    with open(user_like_genre, 'w', encoding='utf-8') as f4:
        json.dump(user_like_genre_dict, f4)

def gen_song_genre():
    with open(song_track,'r',encoding='utf-8') as f:
        song_track_dict = json.load(f)
    song_genre_dict = {}
    with open(train_v,'r',encoding='utf-8') as f2:
        for line in f2:
            user, song, _ = line.strip().split('\t')
            song_genre_dict[song] = {}
            track = song_track_dict[song]
            genre_file = genre_dir + '/' + track + '.json'
            try:
                with open(genre_file,'r',encoding='utf-8') as f3:
                    genre_dict = json.load(f3)
            except:
                continue
            tags = genre_dict['tags']
            tag_dict = dict(tags)
            for k, v in tag_dict.items():
                if float(v) >= 50:
                    song_genre_dict[song][k] = float(v)
    with open(song_genre,'w',encoding='utf-8') as f3:
        json.dump(song_genre_dict,f3)

if __name__ == '__main__':
    pass
    #gen_song_popularity
    #gen_song_artist()
    #gen_user_like()
    #gen_song_track()
    #gen_song_year()
    #gen_track_info()
    #gen_user_like_genre()
    #gen_song_genre()