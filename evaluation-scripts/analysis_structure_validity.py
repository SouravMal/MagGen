


import ase.io
import pandas as pd



# Source of atomic radius : periodictable.com

atomic_radius = {"1":53, "2":31, "3":167, "4":112, "5":87, "6":67, "7":56, "8":48, "9":42, "10":38,\
               "11":190, "12":145, "13":118, "14":111, "15":98, "16":87, "17":79, "18":71, "19":243,\
               "20":194, "21":184, "22":176, "23":171, "24":166, "25":161, "26":156, "27":152, "28":149,\
               "29":145, "30":142, "31":136, "32":125, "33":114,"34":103, "35":94,"36":87, "37":265,\
               "38":219, "39":212, "40":206, "41":198, "42":190, "43":183, "44":178, "45":173, "46":169,\
               "47":165, "48":161, "49":156, "50":145, "51":133, "52":123, "53":115, "54":108, "55":298, "56":253,\
               "57":0, "58":0, "59":247, "60":206, "61":205, "62":238, "63":231, "64":233, "65":225, "66":228,\
               "67":226, "68":226, "69":222, "70":222, "71":217, "72":208, "73":200, "74":193, "75":188, "76":185,\
               "77":180, "78":177, "79":174, "80":171, "81":156, "82":154, "83":143, "84":135, "85":127, "86":120,\
               "87":0, "88":0, "89":0, "90":0, "91":0, "92":0, "93":0, "94":0}



def get_validity(atomic_numbers,distances):
    Nmat = len(atomic_numbers)
    validity = []
    invalid_pairs = []
    for i in range(Nmat):
        for j in range(i+1,Nmat):
            dist = distances[i+j-1]
            atom1 = atomic_numbers[i]
            atom2 = atomic_numbers[j]
            radius_1 = 0.01*atomic_radius[str(atom1)]  # in Angstorm
            radius_2 = 0.01*atomic_radius[str(atom2)] # in Angstorm
            cutoff = radius_1 + radius_2 + 0.25 
            if dist > cutoff :
                validity.append("valid")
            else :
                validity.append("invalid")
                invalid_pairs.append((i+1,j+1,dist))
    count = validity.count("invalid")
    if count == 0 :
        comment = "valid"
    if count <= 5 :
        comment = "moderate"
    if count > 5 :
        comment = "invalid"
    return comment





