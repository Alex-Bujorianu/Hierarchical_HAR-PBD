
import csv
import os
import json
import numpy



def loadalldata(sourcefolder):
    #print(sourcefolder)
    alldata = {}
    
    pfolders = getfoldercontent(sourcefolder, '', 'label')
    pfolders = getfoldercontent(pfolders[0], '', 'label')
    #print(pfolders)

    for p in pfolders:
        #print(p)
        pfoldernamesplit = os.path.split(p)
        pfoldernamesplit = pfoldernamesplit[1].split("_")
        #print(pfoldernamesplit)
        pid = int(pfoldernamesplit[0][1:])
        #print(pid)
        participantdata = getparticipantdata(p)
        if len(participantdata) != 0: alldata[str(pid)] = participantdata

    print("all data now loaded")    
    print("")

    #print(alldata)

    return alldata
        
    

def getparticipantdata(participantfolderpath):

    participantdata = {}

    sessfolders = getsessionfolders(participantfolderpath)

    for s in sessfolders:

        #print(s)
        
        datafolders = getdatafolders(s)

        for d in datafolders:

            #print(d)

            ddata, ddatadatetime = getfulldata(d)
            if len(ddata) != 0: participantdata[ddatadatetime] = ddata


    return participantdata


def getsessionfolders(participantfolderpath):

    psubfolders = getfoldercontent(participantfolderpath, 'movement', '')

    for psubf in psubfolders:

        return getfoldercontent(psubf, '', '')


def getdatafolders(sessionfolderpath):

    datafolders = []

    psesssubfolders = getfoldercontent(sessionfolderpath, '', '')

    for psesssubf in psesssubfolders:

        #print(psesssubf)

        datafolders.extend(getfoldercontent(psesssubf, 'csv', ''))

    return datafolders

                    
def getfoldercontent(folderpath, inclusion, exclusion):

    content = []

    if not os.path.isdir(folderpath): return content
            
    for c in os.listdir(folderpath):

        fits = True

        if inclusion!='' and (inclusion.lower() not in c.lower()): fits = False
        if exclusion!='' and (exclusion.lower() in c.lower()): fits = False 
            
        if fits: content.append(os.path.join(folderpath, c))

    return content




def getfulldata(datafolderpath):

    fulldata = {}

    meta = getmeta(datafolderpath)

    if len(meta) == 0: return fulldata, None
    
    year, month, day, hour, minute, second = getstarttime(meta)
    samplingrate = float(meta['frequency'])
    framecount = int(meta['frame_count'])
    fulldata['year'] = year
    fulldata['month'] = month
    fulldata['day'] = day
    fulldata['hour'] = hour
    fulldata['minute'] = minute
    fulldata['second'] = second
    fulldata['samplingrate'] = samplingrate
    fulldata['framecount'] = framecount
    fulldatakey = '_'.join([str(year), str(month), str(day), str(hour), str(minute)])
    
    dfiles = getfoldercontent(datafolderpath, '', 'meta')
    for file in dfiles:
        data, bone, isdata = getbonedata(file, meta)
        if isdata: fulldata[bone] = data

    return fulldata, fulldatakey


def getmeta(datafolderpath):

    meta = {}

    if os.path.getsize(os.path.join(datafolderpath, 'meta.json')) == 0: return meta

    with open(os.path.join(datafolderpath, 'meta.json')) as metafile:


        meta = json.load(metafile)
        
    return meta


def getstarttime(meta):
    
    datetimecode = meta['measured']
    datetimecodesplit = datetimecode.split('T')
    datesplit = datetimecodesplit[0].split('-')
    year = int(datesplit[0])
    month = int(datesplit[1])
    day = int(datesplit[2])
    
 
    timesplit = datetimecodesplit[1].split(':')
    hour = int(timesplit[0])
    minute = int(timesplit[1])
    secondandmilli = timesplit[2].split('+')
    secondandmillisplit = secondandmilli[0].split('.')
    millisecond = float(secondandmillisplit[1][:3])    
    second = float(secondandmillisplit[0]) + millisecond/1000.0

    return year, month, day, hour, minute, second
     

def getbonedata(fullfilename, meta):

    dataarr  = numpy.array([])
    isbonedata = False

    fullfilenamesplit = os.path.split(fullfilename)
    filenamesplit = fullfilenamesplit[1].split('.')
    fileext = filenamesplit[1]
    
    filenamesplit = filenamesplit[0].split('_')
    angleorpos = filenamesplit[0]
    bone = filenamesplit[1]

    
    if fileext == 'csv' and angleorpos.lower()=='positions' and isinbones(meta['bones'], bone):

        isbonedata = True

        with open(fullfilename, 'r') as csvfile:

            data = []         
            rowcount = 0
            myreader = csv.reader(csvfile)

            for row in myreader:

                if rowcount >= 1:

                    valcount = 0
                    temp = []

                    for val in row:


                        if valcount >=1 and valcount <=3:
                            temp.append(float(val))

                        valcount+=1 


                    data.append(temp)

                rowcount+=1
            dataarr = numpy.array(data)

            #print(dataarr.shape)

    return dataarr, bone, isbonedata


def isinbones(bonelist, bone):

    for b in bonelist:

        if b==bone: return True

    return False
    
        
def loadalllabels(sourcefile):

    with open(sourcefile, 'r') as csvfile:

        labels = {}
        year = 2021

        rowcount = 0
        myreader = csv.reader(csvfile)

        for row in myreader:

            if rowcount >= 1:

                valcount = 0
                painworryconfidence = []
                instance = {}

                for val in row:

                    if valcount == 0:
                        pid = int(val)

                    elif valcount == 2:
                        activity = int(val)

                    elif valcount == 4:
                        challenging = False
                        if val.lower() == 'yes': challenging = True

                    elif valcount == 5:
                        day = int(val)
                         
                    elif valcount == 6:
                        month = int(val)
                     
                    elif valcount == 7:
                        hour = int(val)

                    elif valcount == 8:
                        minute = int(val)

                    elif valcount == 9:
                        painworryconfidence.append(float(val))

                    elif valcount == 10:
                        painworryconfidence.append(float(val))

                    elif valcount == 11:
                        painworryconfidence.append(float(val))

                    elif valcount == 12:
                        sequence = int(val)
                     
                    valcount += 1

                instancekey = '_'.join([str(pid), str(year), str(month), str(day), str(hour), str(minute)])
                instance['activity'] = activity
                instance['challenging'] = challenging
                instance['labels'] = painworryconfidence
                instance['sequenceid'] = (pid*20)+sequence
                labels[instancekey] = instance

            rowcount += 1


        
    
    print("all labels now loaded")
    print("")

    #print(labels)

    return labels
                 


    





        
