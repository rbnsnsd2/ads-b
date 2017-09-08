#!/usr/bin/env python
# -*- coding: latin-1 -*-
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv, os, pickle
import datetime as dt
#import scipy.interpolate as inter

## Change the wording directory to the location of the source documents
os.chdir('/Users/rbnsnsd2/Documents/Python/ADSB/')



def e_time(date,time):
    """Take the date and time values and convert to epoch in seconds.

    Input:
        - date, time lists
    Output:
        - 1d list
    """
    epoch = dt.datetime.utcfromtimestamp(0)
    T = dt.datetime.combine(dt.datetime.strptime(date, "%Y/%m/%d"),
                            dt.datetime.strptime(time, "%H:%M:%S.%f").time())
    E = T - epoch
    return E.total_seconds()

def csv_extract_col(csvinput):
    """
    Extract the id and state information as lists
    """
    with open(csvinput, 'r', encoding='latin1') as csvfile:
        datareader = csv.reader(csvfile, delimiter = ',')
        Code, Date, Time, Alt, GS, Lat, Long = [], [], [], [], [], [], []
        for ind,row in enumerate(datareader):
            if len(row)==22: 
                try:
                    for index,col in enumerate(row):
                        if (index==4 and not col==''):   Code.append(col)
                        if (index==6 and not col==''):   Date.append(col)
                        if (index==7 and not col==''):   Time.append(col)                    
                        if index==11: Alt.append(float(col)) if not col=='' else Alt.append(np.float('NaN'))
                        if index==12: GS.append(float(col)) if not col=='' else GS.append(np.float('NaN')) 
                        if index==14: Lat.append(float(col)) if not col=='' else Lat.append(np.float('NaN'))                   
                        if index==15: Long.append(float(col)) if not col=='' else Long.append(np.float('NaN'))
                except ValueError: print(ind,index,col)
            else: pass
    return Code, Date, Time, Alt, GS, Lat, Long
 
#BKN dictionary
AircraftDict = {'BKN28':'A8353D', 'BKN24':'A37E74', 'BKN29':'A39107',
       'BKN23':'A5080C','BKN20':'A81785','BKN25':'A3822B','BKN47':'A24B48',
       'BKN26':'A51331','BKN27':'A38999','BKN48':'A24EFF', 'BKN21':'A81B3C'}

#Column index value dictionary
ColumnIndex = {'Code':4,'Time':7,'Ident':10,'Alt':11,'GS':12,'TC':13,
               'Lat':14,'Long':15,'VS':16,'Squawk':17}
               
#sourcefile = 'BKN25.csv'
#sourcefile = 'BKN47.csv'
#sourcefile = 'BKN20_1.csv'
#sourcefile = '09-17-15.csv'
sourcefile = '09-13-16.txt'


Code, Date, Time, Alt, GS, Lat, Long = csv_extract_col(sourcefile)
print('csv extract done')
#Etime = [e_time(D,T) for D,T in zip(Date,Time)]
#print('etime calculated')

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def lin_interp(time,ordinate):
    T = np.array(time)
    O = np.array(ordinate)
    nans, y = nan_helper(O)
    return np.interp(T,T[~nans],O[~nans])

#Filter out the aircraft that you want
class Flight:
    def __init__(self,ident,hexcodelist):
        self.ident = ident
        try: 
            self.hex = AircraftDict[ident]
        except: self.hex = self.ident
        self.index = [i for i,j in enumerate(hexcodelist) if j==self.hex]
        
    def add_etime(self,Date,Time):
        self.date = [Date[i] for i in self.index]
        self.time = [Time[i] for i in self.index]
        self.epochtime = [e_time(D,T) for D,T in zip(self.date,self.time)]
    def add_alt_gs(self,alt,gs):
        Alt = [alt[i] for i in self.index]
        self.altitude = lin_interp(self.epochtime,Alt)
        GS = [gs[i] for i in self.index]
        self.groundspeed = lin_interp(self.epochtime,GS)
    def add_latlong(self,lat,long):
        Lat = [lat[i] for i in self.index]
        self.latitude = lin_interp(self.epochtime,Lat)
        Long = [long[i] for i in self.index]
        self.longitude = lin_interp(self.epochtime, Long)
    def add_grid(self,origin_lat,origin_long):
        self.dx = (self.longitude - origin_long)*40000*np.cos((self.latitude + origin_lat)*np.pi/360)/360
        self.dy = (origin_lat - self.latitude)*40000/360
        

#import math
#dx = (lon2-lon1)*40000*math.cos((lat1+lat2)*math.pi/360)/360
#dy = (lat1-lat2)*40000/360


#pickle.dump(data1,open("ASRSdataQ.pkl", "wb"))

#KCPS Runway data Elevation 413MSL
RWYLat = [38.573884,38.568805,38.563900]
RWYLong = [-90.164848,-90.154519,-90.144627]
TDZLong = [-90.161839]
TDZLat = [38.572358]
FNLLong = [-90.179997]
FNLLat = [38.581354] #713
RWY12lLong = [-90.158010,-90.149880,-90.142953]
RWY12lLat = [38.573690,38.569674,38.566227]
RWY5Long = [-90.168398,-90.165288,-90.161522]
RWY5Lat = [38.572469,38.574775,38.577483]

RWY12R_tdz = [38.572377,-90.161788]
RWY12R_app_path = [38.581030,-90.179060,300] 

class Airport:
    def __init__(self,airportID,airportCenter):
        self.airport_id, self.airport_center = airportID, airportCenter
    def getName(self):
        return self.name
        
class Rwy(Airport):
    def __init__(self,airportID,airportCenter,runwayID,tdzElev):
#    def __init__(self,runwayid):
        super(self.__class__, self).__init__(airportID,airportCenter)
        self.runway_id = runwayID
        self.tdz_elev = tdzElev
    def tdz(self,latLong):
        self.tdz_lat_long = latLong
    def app_path(self,latLongAgl):
        self.app_path = latLongAgl
    def threshold(self,latLong):
        self.thresh_lat_long = latLong
    def depend(self,latLong):
        self.dep_lat_long = latLong
#    def add_grid(self,origin_lat,origin_long):
#        self.dx = (self.longitude - origin_long)*40000*np.cos((self.latitude + origin_lat)*np.pi/360)/360
#        self.dy = (origin_lat - self.latitude)*40000/360
        
#RWY30L = Rwy('KCPS',RWY12R_tdz,'30L')
#RWY30L.tdz([TDZLat,TDZLong])
#RWY30L.app_path([TDZLat,TDZLong,400])   
RWY12R = Rwy('KCPS',RWY12R_tdz,'12R')
RWY12R.tdz(RWY12R_tdz)
RWY12R.app_path(RWY12R_app_path)        


class Approach:
    def __init__(self):
        self.qual = 'NaN'

#set origin at tdz then c=0
#m=y/x   
#y=mx+c
#dx = (self.longitude - origin_long)*40000*np.cos((self.latitude + origin_lat)*np.pi/360)/360
#dy = (origin_lat - self.latitude)*40000/360

#x=(y-c)/m
#self.dx = (self.longitude - origin_long)*40000*np.cos((self.latitude + origin_lat)*np.pi/360)/360
#self.dy = (origin_lat - self.latitude)*40000/360

 
BKN23 = Flight('BKN23',Code)
BKN23.add_etime(Date,Time)
BKN23.add_alt_gs(Alt,GS)
BKN23.add_latlong(Lat,Long)
BKN23.add_grid(TDZLat,TDZLong)

#An individual flight is BKN23.epochtime[26000:] which has index 245678
#in the Time list
#2016/09/13 14:53 KCPS 131453Z VRB04KT 10SM CLR 24/16 A3013 RMK AO2 SLP200

#PA to AGL
#PA = 145366*(1-(Pressure/1013.25)**0.190284)
#Alt + PA = MSL
#tdzAGL = Alt + PA - tdz_elev


# setup mercator map projection.
def kcps_figure(name,Long,Lat,Alt,GS):
    plt.clf()
    fig=plt.figure()
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    #m = Basemap(llcrnrlon=-90.182,llcrnrlat=38.55,urcrnrlon=-90.125,urcrnrlat=38.58,\
    #        rsphere=(6378137.00,6356752.3142),\
    #        resolution='l',projection='merc',\
    #        lat_0=40.,lon_0=-20.,lat_ts=20.)
#    m = Basemap(llcrnrlon=-90.23,llcrnrlat=38.545,urcrnrlon=-90.1,urcrnrlat=38.615,\
#            rsphere=(6378137.00,6356752.3142),\
#            resolution='l',projection='merc',\
#            lat_0=40.,lon_0=-20.,lat_ts=20.)
#    m = Basemap(llcrnrlon=-90.33,llcrnrlat=38.445,urcrnrlon=-90.0,urcrnrlat=38.675,\
#        rsphere=(6378137.00,6356752.3142),\
#        resolution='l',projection='merc',\
#        lat_0=40.,lon_0=-20.,lat_ts=20.)
    m = Basemap(llcrnrlon=-90.43,llcrnrlat=38.345,urcrnrlon=-89.9,urcrnrlat=38.775,\
        rsphere=(6378137.00,6356752.3142),\
        resolution='l',projection='merc',\
        lat_0=40.,lon_0=-20.,lat_ts=20.)
    C = matplotlib.colors.Normalize()
    m.plot(RWYLong,RWYLat, color="red", linewidth=2, latlon=True)
    m.plot(RWY12lLong,RWY12lLat, color="red", linewidth=2, latlon=True)
    m.plot(RWY5Long,RWY5Lat, color="red", linewidth=2, latlon=True)
    C = matplotlib.colors.Normalize(Alt)
    CC = matplotlib.colors.Colormap(C)
    col = []
    AA = Alt/1400
    G = GS/110
    m.scatter(Long,Lat, c= plt.cm.jet(G), linewidth=0, latlon=True)
#    m.scatter(Long,Lat, c= plt.cm.jet(AA), linewidth=0, latlon=True)
#    m.drawstates()
    #m.fillcontinents()
    m.drawrivers()
    ax.set_title(name)
    plt.savefig(str(name)+'.jpg',dpi=288)
    plt.show()
    #    return

#kcps_figure('BKN23',BKN23.longitude,BKN23.latitude,BKN23.altitude,BKN23.groundspeed)
