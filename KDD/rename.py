import os

def rename_img():
    
    i = 0
    for filename in os.listdir("/home/johnomena/data-min/sadness/"): 
        dst ="sadness." + str(i) + ".jpg"
        src ='/home/johnomena/data-min/sadness/'+ filename 
        dst ='/home/johnomena/data-min/sadness/'+ dst 
        os.rename(src, dst) 
        i += 1

rename_img()