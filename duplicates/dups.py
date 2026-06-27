from collections import defaultdict
import os
dictionary = defaultdict()
toremoveDictionary = []
def isDuplicate(file1, file2):
    result = True
    f1 = open(file1)
    f2 = open(file2)

    f1_data = f1.readlines()
    f2_data = f2.readlines()

    i = 0

    for line1, line2 in zip(f1_data, f2_data):
        i += 1
        if line1 == line2:
            print(f"Line {i}: IDENTICAL")
        else:
            print(f"Line {i}:")
            print(f"\tFile 1: {line1.strip()}")
            print(f"\tFile 2: {line2.strip()}")
            result = False

    f1.close()
    f2.close()
    
    return result
def returnOlderFile(file1, file2):
    
    result = 0
    # Get modification times (in seconds since epoch)
    time1 = os.stat(file1).st_mtime
    time2 = os.stat(file2).st_mtime

    if time1 < time2:
        result = 0
        print(f"{file1} is older than {file2}")
    elif time1 > time2:
        print(f"{file2} is older than {file1}")
        result = 1
           
    return result


myvars = {}
with open("duplicates/reff.cnf") as myfile:
    for line in myfile:
        name, var = line[:-1].strip().partition("=")[::2]
        myvars[name.strip()] = var.strip()



print(myvars)

#create catogories by size of the file. 


for path, subdirs, files in os.walk(myvars["rootFolder"]):
    for name in files:
        
        #print(os.path.join(path, name))
        file_stats = os.stat(os.path.join(path, name))
        #print(file_stats.st_size)
        dictionary.setdefault(file_stats.st_size, []).append(os.path.join(path, name))



for element in dictionary:
    if len(dictionary[element]) > 1:
        #
        # get list of potentially duplicated files
        # compare in loop. reduce list to unique files. 
        for k in range(len(dictionary[element])):
          print("we are in pool  I:", k, "Value:", k)
          for j in range(len(dictionary[element])):
            if k!=j and dictionary[element][j]!=-1 and j>k:
                if isDuplicate(dictionary[element][k],dictionary[element][j]):
                    
                    if returnOlderFile(dictionary[element][k],dictionary[element][j]) == 0:
                        toremoveDictionary.append(dictionary[element][j])
                        dictionary[element][j]=-1  ## put also into dictionary path that can be removed
                    else:
                        toremoveDictionary.append(dictionary[element][k])
                        dictionary[element][k]=-1  ## put also into dictionary path that can be removed
                    
                

print('potentially unique')        
print(dictionary)
print('potentially toremove')
print(toremoveDictionary)
