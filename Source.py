from textHandler import HamshahriReader
hamshahri = HamshahriReader(root='Corpus')
alllabel=[]
i=1
DIR='data/Hamshahri/'
for doc in hamshahri.texts():
    with open(DIR+'/'+str(i)+'.txt','w') as f:
        f.write(doc[0])
    with open(DIR+'/'+str(i)+'.lab','w') as f:
        f.write(doc[1])
    alllabel.append(doc[1])
    i+=1
    if i==15000:
        break

alllabel=list(set(alllabel))
print(len(alllabel))
print(alllabel)
with open('/home/apasai/PycharmProjects/textClassificationWithMagpie/data/Hamshahri.labels','w') as file:
    for i in alllabel:
        file.write(i+'\n')

