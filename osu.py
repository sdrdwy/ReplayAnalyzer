#osu file

#Simple Class for saving structred osufile
class Osu:
    def __init__(self):
        self.raw=[]
        self.data={}
        self.data["[General]"]={}
        self.data["[Editor]"]={}
        self.data["[Metadata]"]={}
        self.data["[Difficulty]"]={}
        self.data["[Events]"]={}
        self.data["[TimingPoints]"]=[]
        self.data["[HitObjects]"]=[]
        self.path=""
        self.tbs=""
    def load_from_file(self,path=""):
        file=open(path,'r',encoding="utf-8")
        self.path=path
        self.raw=file.readlines()
        self.version = self.raw[0][self.raw[0].find("v"):].replace("\n","")
        index=0
        while self.raw[index]!="[General]\n":
            index+=1
        #-- load general --#
        while self.raw[index]!="[Editor]\n":
            begin=self.raw[index].find(":")
            if begin != -1 :
                self.data["[General]"][self.raw[index][:begin]]=\
                    self.raw[index][begin+1:].replace("\n","").strip()
            index+=1
            
        #-- load Editor --#
        while self.raw[index]!='[Metadata]\n':
            begin=self.raw[index].find(":")
            if begin != -1 :
                self.data["[Editor]"][self.raw[index][:begin]]=\
                    self.raw[index][begin+1:].replace("\n","").strip()
            index+=1
        #-- load Metadata --#
        while self.raw[index]!='[Difficulty]\n':
            begin=self.raw[index].find(":")
            if begin != -1 :
                self.data["[Metadata]"][self.raw[index][:begin]]=\
                    self.raw[index][begin+1:].replace("\n","").strip()
            index+=1
        #-- load difficulty --#
        while self.raw[index]!='[Events]\n':
            begin=self.raw[index].find(":")
            if begin != -1 :
                self.data["[Difficulty]"][self.raw[index][:begin]]=self.raw[index][begin+1:].replace("\n","")
            index+=1
        #-- load Events --#
        self.data["[Events]"]["raw"]=''
        while self.raw[index]!='[TimingPoints]\n':
            index+=1
            if "[" not in self.raw[index]:
                self.data["[Events]"]["raw"]+=self.raw[index]
            
        #-- load TimingPoints --#
        while self.raw[index]!='[HitObjects]\n':
            index+=1
            if self.raw[index]=='[HitObjects]\n':
                break
            if self.raw[index].strip() == "":
                continue
            l=self.raw[index].split(",")
            if len(l) < 2:
                continue
            if(len(l)==8):
                l[7]=l[7].replace("\n","")
            self.data["[TimingPoints]"].append(l)
        #-- load HitObjects --#
        for i in range(index+1,len(self.raw)):
            if self.raw[i].strip() == "":
                continue
            l=self.raw[i].split(",")
            if len(l) < 3:
                continue
            if(len(l)==6):
                s=l[5].replace("\n","")
                l[5]=s[:s.find(":")]
                l.append(s[s.find(":"):])
            self.data["[HitObjects]"].append(l)
        file.close()
    def output(self,tilte=""):
        outstr=tilte+"\n"
        if(tilte == "[Events]"):
            outstr+=self.data[tilte]["raw"]
            return outstr
        if(tilte == "[TimingPoints]"):
            for i in self.data[tilte]:
                s=','.join(i)
                outstr+=s
                if "\n" not in s:
                    outstr+='\n'
            return outstr
        if(tilte == "[HitObjects]"):
            for i in self.data[tilte]:
                if(len(i)>2):
                    s=','.join(i[:6])
                    s+=i[6]
                else:
                    s=','.join(i)
                outstr+=s
                if "\n" not in s:
                    outstr+='\n'
            return outstr
        for k,v in self.data[tilte].items():
            if tilte in ["[Metadata]","[Difficulty]"]:
                outstr+=k+":"+v
            else:
                outstr+=k+": "+v
            if "\n" not in v:
                outstr+="\n"
        return outstr
    def outfile(self,path=""):
        outstr="osu file format "+self.version+"\n\n"
        outstr+=self.output("[General]")+"\n"
        outstr+=self.output("[Editor]")+"\n"
        outstr+=self.output("[Metadata]")+"\n"
        outstr+=self.output("[Difficulty]")+"\n"
        outstr+=self.output("[Events]")+""
        outstr+=self.output("[TimingPoints]")+""
        outstr+=self.output("[HitObjects]")+"\n\n"
        f=open(path,"w",encoding="utf-8")
        f.write(outstr)
        f.close()
        return outstr



if __name__=="__main__":
    f=Osu()
    #testing
    f.load_from_file("E:\\osumap\\osu!\\Songs\\2049944 Bambinton - Zaya\\Bambinton - Zaya (YuEast 2018) [Northern Story.].osu")
    print(f.output("[HitObjects]"))
    f.data["[Metadata]"]["BeatmapID"]='-1'
    f.outfile("3.txt")
