#osu file

import math
from osu import Osu
class Mania:
    def __init__(self,file):
        self.file=file
        self.osu=Osu()
        self.osu.load_from_file(file)
        self.cs=int(self.osu.data["[Difficulty]"]["CircleSize"])
        self.keys=self.osu.data["[HitObjects]"]
        self.key_index={}
        for _ in range(self.cs):
            self.key_index[_]=[]
        for i in range(len(self.keys)):
            try:
                index=math.floor(int(self.keys[i][0])*self.cs/512)
            except (ValueError, IndexError):
                continue
            if 0 <= index < self.cs:
                self.key_index[index].append(i)
    def get_key(self,key,index): #the 2nd note in key0-> key=0,index=1#
        return self.keys[self.key_index[key][index]]
    def set_key(self,key,index,newkey):
        self.keys[self.key_index[key][index]]=newkey
    def save_file(self,path):
        self.osu.data["[HitObjects]"]=self.keys
        self.osu.outfile(path)
if __name__=="__main__":
    #testing
    m=Mania("E:\\osumap\\osu!\\Songs\\2049944 Bambinton - Zaya\\Bambinton - Zaya (YuEast 2018) [Northern Story.].osu")
    print(m.cs)
    print(m.keys[m.key_index[0][-1]])
    '''k=m.get_key(0,-1)
    k[3]='1'
    m.set_key(0,-1,k)
    m.save_file("5.txt")'''
    print(m.get_key(0,-1))