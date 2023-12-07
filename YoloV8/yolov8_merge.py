import os
import shutil

train_to_merge = [1,6]
video = "vg_3"

files_to_merge = [f"./RealTimeMOT/YoloV8/Train{i}/{video}/det" for i in train_to_merge]
merge = "Merge"
for i in train_to_merge :
  merge += f"_{i}"

os.mkdir(f"./RealTimeMOT/YoloV8/{merge}")
os.mkdir(f"./RealTimeMOT/YoloV8/{merge}/{video}")
shutil.copyfile(f"./RealTimeMOT/YoloV8/Train{train_to_merge[0]}/{video}/seqinfo.ini",f"./RealTimeMOT/YoloV8/{merge}/{video}/seqinfo.ini")
shutil.copytree(f"./RealTimeMOT/YoloV8/Train{train_to_merge[0]}/{video}/img1",f"./RealTimeMOT/YoloV8/{merge}/{video}/img1")

os.mkdir(f"./RealTimeMOT/YoloV8/{merge}/{video}/det")
det = open(f"./RealTimeMOT/YoloV8/{merge}/{video}/det/det.txt","a")

det_to_merge = []
index_det_to_merge = [0 for i in train_to_merge]
len_det_to_merge = []
for i in range(len(train_to_merge)) :
  file = open(f"./RealTimeMOT/YoloV8/Train{train_to_merge[i]}/{video}/det/det.txt","r")
  det_to_merge.append(file.readlines())
  len_det_to_merge.append( (lambda f : sum(1 for line in f))(det_to_merge[i]) )
  file.close()

seq_info = open(f"./RealTimeMOT/YoloV8/{merge}/{video}/seqinfo.ini","r")
seqlength = int(seq_info.readlines()[4][10:])
seq_info.close()

for frame in range(1,seqlength+1) :
  for i in range(len(train_to_merge)) :
    det_file = det_to_merge[i]
    if index_det_to_merge[i] < len_det_to_merge[i] :
      data = det_file[index_det_to_merge[i]].split(",")
      while index_det_to_merge[i] < len_det_to_merge[i] and int(data[0]) == frame :
        line = det_file[index_det_to_merge[i]]
        det.write(line)
        
        index_det_to_merge[i] += 1
        if index_det_to_merge[i] < len_det_to_merge[i] :
          data = det_file[index_det_to_merge[i]].split(",")
  

det.close()