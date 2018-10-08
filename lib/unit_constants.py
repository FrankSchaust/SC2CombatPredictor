#ids
terran_ids = [49,#reaper
              51,#Marauder
              48,#Marine
              50,#Ghost
              268,#Mule
              45,#SCV
              56,#Raven
              55,#Banshee
              54,#Medivac
              34,35,#Viking
              57,#Battlecruiser
              52,#Thor
              32,33,#Siege Tank
              53,484,#Hellion
              692#Cyclone
              ]
protoss_ids = [10,#Mutterschiff 
               80,#Void Ray 
               76,#Dark Templar 
               79,#Carrier 
               4,#Colossus 
               73,#Zealot 
               84,#Probe 
               78,#Phoenix 
               141,#Archon 
               83,#Immortal 
               82,#Observer 
               75,#High Templar
               74,#Stalker
               694,#Disruptor
               77,#Sentry
               85,#Interceptor
              311#Adept
              ]

zerg_ids = [111,#Verseucher
            112,#Schänder
            129,#Overseer
            105,#Zergling 
            114,#Brutlord
            107,#Hydralisk
            126,#Königin
            108,#Mutalisk
            109,#Ultralisk
            110,#Roach//Schabe
            104,#Drohne
            106,#Overlord
            13,#Formling_Zealot
            17,#Formling_Zergling
            15,#Formling_Marine
            9,#Berstling
            289,#Brütling
            499,#Viper
            12#Formling
            ]
    

### Constants for terran units
reaper = {
    'attributes': ['b', 'l'],
    'pw_g': 10.1,
    'pw_a': 0,
    'hp': 60,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 50,
    'gas': 50,
    'supply': 1
    }
marauder = {
    'attributes': ['a', 'b'],
    'pw_g': 9.3,
    'pw_a': 0,
    'hp': 125,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 9.3, 'g'],
    'min': 100,
    'gas': 25,
    'supply': 2
    }
marine = {
    'attributes': ['b', 'l'],
    'pw_g': 9.8,
    'pw_a': 9.8,
    'hp': 45,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'min': 50,
    'gas': 0,
    'supply': 1
    }
ghost = {
    'attributes': ['b', 'p'],
    'pw_g': 9.3,
    'pw_a': 9.3,
    'hp': 100,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'bonus': ['l', 9.3, 'ga'],
    'min': 150,
    'gas': 125,
    'supply': 2
    }
mule = {
    'attributes': ['l', 'm'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 60,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'n',
    'det': 'n',
    'inv': 'n',
    'min': 0,
    'gas': 0,
    'supply': 0
    }
scv = {
    'attributes': ['b', 'l', 'm'],
    'pw_g': 4.67,
    'pw_a': 0,
    'hp': 45,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 50,
    'gas': 0,
    'supply': 1
    }
raven = {
    'attributes': ['l', 'm', 'd'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 140,
    'sh': 0,
    'ar':1,
    'type': 'a',
    'target': 'n',
    'det': 'y',
    'inv': 'n',
    'min': 100,
    'gas': 200,
    'supply': 2
    }
banshee = {
    'attributes': ['l', 'm'],
    'pw_g': 27,
    'pw_a': 0,
    'hp': 140,
    'sh': 0,
    'ar':0,
    'type': 'a',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 150,
    'gas': 100,
    'supply': 3
    }
medivac = {
    'attributes': ['a', 'm'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 150,
    'sh': 0,
    'ar':1,
    'type': 'a',
    'target': 'n',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 100,
    'supply': 2
    }
viking = {
    'attributes': ['a', 'm'],
    'pw_g': 0,
    'pw_a': 14,
    'hp': 135,
    'sh': 0,
    'ar':0,
    'type': 'a',
    'target': 'a',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 5.6, 'a'],
    'min': 150,
    'gas': 75,
    'supply': 2
    }
battlecruiser = {
    'attributes': ['a', 'm', 'massive'],
    'pw_g': 50,
    'pw_a': 37.5,
    'hp': 550,
    'sh': 0,
    'ar':3,
    'type': 'a',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'min': 400,
    'gas': 300,
    'supply': 6
    }
thor = {
    'attributes': ['a', 'm', 'massive'],
    'pw_g': 65.9,
    'pw_a': 11.2,
    'hp': 400,
    'sh': 0,
    'ar':2,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'bonus': ['l', 11.2, 'a'],
    'min': 300,
    'gas': 200,
    'supply': 6
    }
siege_tank = {
    'attributes': ['a', 'm'],
    'pw_g': 20.27,
    'pw_a': 0,
    'hp': 175,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 13.51],
    'min': 150,
    'gas': 125,
    'supply': 3
    }
hellion = {
    'attributes': ['l', 'm'],
    'pw_g': 4.48,
    'pw_a': 0,
    'hp': 90,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['l', 3.4],
    'min': 100,
    'gas': 0,
    'supply': 2
    }
cyclone = {
    'attributes': ['a', 'm'],
    'pw_g': 30,
    'pw_a': 0,
    'hp': 180,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 20, 'g'],
    'min': 150,
    'gas': 100,
    'supply': 3
    }

### Constants for Protoss Units
mothership = {
    'attributes': ['a', 'massive', 'm', 'p'],
    'pw_g': 22.8,
    'pw_a': 22.8,
    'hp': 350,
    'sh': 350,
    'ar':2,
    'type': 'a',
    'target': 'ga',
    'det': 'n',
    'inv': 'y',
    'min': 400,
    'gas': 400,
    'supply': 8
    }
void_ray = {
    'attributes': ['a', 'm'], 
    'pw_g': 16.8,
    'pw_a': 16.8,
    'hp': 150,
    'sh': 100,
    'ar':0,
    'type': 'a',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 11.2, 'ga'],
    'min': 250,
    'gas': 150,
    'supply': 4
    }
dark_templar = {
    'attributes': ['b', 'l', 'p'],
    'pw_g': 37.2,
    'pw_a': 0,
    'hp': 40,
    'sh': 80,
    'ar': 1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'y',
    'min': 125,
    'gas': 125,
    'supply': 2
    }
carrier = {
    'attributes': ['a', 'massive', 'm'],
    'pw_g': 37.4,
    'pw_a': 37.4,
    'hp': 250,
    'sh': 150,
    'ar':2,
    'type': 'a',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'min': 350,
    'gas': 250,
    'supply': 6
    }
colossus = {
    'attributes': ['a', 'massive', 'm'],
    'pw_g': 18.7,
    'pw_a': 0,
    'hp': 200,
    'sh': 150,
    'ar':1,
    'type': 'ga',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['l', 9.3, 'g'],
    'min': 300,
    'gas': 200,
    'supply': 6
    }
zealot = {
    'attributes': ['l', 'b'],
    'pw_g': 18.6,
    'pw_a': 0,
    'hp': 100,
    'sh': 50,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 0,
    'supply': 2
    }
probe = {
    'attributes': ['l', 'm'],
    'pw_g': 4.67,
    'pw_a': 0,
    'hp': 20,
    'sh': 20,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 50,
    'gas': 0,
    'supply': 1
    }
phoenix = {
    'attributes': ['l', 'm'],
    'pw_g': 0,
    'pw_a': 12.7,
    'hp': 120,
    'sh': 60,
    'ar':0,
    'type': 'a',
    'target': 'a',
    'det': 'n',
    'inv': 'n',
    'bonus': ['l', 12.7, 'a'],
    'min': 150,
    'gas': 100,
    'supply': 2
    }
archon = {
    'attributes': ['massive', 'p'],
    'pw_g': 20,
    'pw_a': 20,
    'hp': 10,
    'sh': 350,
    'ar':0,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'bonus': ['b', 8, 'ga'],
    'min': 250,
    'gas': 250,
    'supply': 4
    }
immortal = {
    'attributes': ['a', 'm'],
    'pw_g': 19.2,
    'pw_a': 0,
    'hp': 200,
    'sh': 100,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 28.9, 'g'],
    'min': 250,
    'gas': 100,
    'supply': 4
    }
observer = {
    'attributes': ['l', 'm', 'd'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 40,
    'sh': 20,
    'ar':0,
    'type': 'a',
    'target': 'n',
    'det': 'y',
    'inv': 'y',
    'min': 25,
    'gas': 75,
    'supply': 1
    }
high_templar = {
    'attributes': ['b', 'l', 'p'],
    'pw_g': 3.2,
    'pw_a': 0,
    'hp': 40,
    'sh': 40,
    'ar': 0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 50,
    'gas': 150,
    'supply': 2
    }
stalker = {
    'attributes': ['a', 'm'],
    'pw_g': 9.7,
    'pw_a': 9.7,
    'hp': 80,
    'sh': 80,
    'ar':1,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 3.7, 'ga'],
    'min': 125,
    'gas': 50,
    'supply': 2
    } 
disruptor = {
    'attributes': ['a', 'm'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 100,
    'sh': 100,
    'ar':1,
    'type': 'g',
    'target': 'n',
    'det': 'n',
    'inv': 'n',
    'min': 150,
    'gas': 150,
    'supply': 3
    }
sentry = {
    'attributes': ['p', 'm', 'l'],
    'pw_g': 8.4,
    'pw_a': 8.4,
    'hp': 40,
    'sh': 40,
    'ar':1,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'min': 50,
    'gas': 100,
    'supply': 2
    }
adept = {
    'attributes': ['l', 'b'],
    'pw_g': 6.2,
    'pw_a': 0,
    'hp': 70,
    'sh': 70,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['l', 7.45, 'g'],
    'min': 100,
    'gas': 25,
    'supply': 2
    }
###Constants for Zerg Units   
drone = {
    'attributes': ['l', 'b'],
    'pw_g': 4.67,
    'pw_a': 0,
    'hp': 40,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 50,
    'gas': 0,
    'supply': 1
    }
queen = {
    'attributes': ['p', 'b'],
    'pw_g': 11.2,
    'pw_a': 12.6,
    'hp': 175,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'min': 150,
    'gas': 0,
    'supply': 2
    }
zergling = {
    'attributes': ['l', 'b'],
    'pw_g': 10,
    'pw_a': 0,
    'hp': 35,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 25,
    'gas': 0,
    'supply': 0.5
    }
baneling = {
    'attributes': ['b'],
    'pw_g': 20,
    'pw_a': 0,
    'hp': 1,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus':['l', 15, 'g'],
    'min': 50,
    'gas': 25,
    'supply': 0.5
    }
roach = {
    'attributes': ['a', 'b'],
    'pw_g': 11.2,
    'pw_a': 0,
    'hp': 145,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 75,
    'gas': 25,
    'supply': 2
    }
ravager = {
    'attributes': ['b'],
    'pw_g': 14.4,
    'pw_a': 0,
    'hp': 120,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 100,
    'supply': 3
    }
hydralisk = {
    'attributes': ['l', 'b'],
    'pw_g': 22.4,
    'pw_a': 22.4,
    'hp': 90,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 50,
    'supply': 2
    }
lurker = {
    'attributes': ['a', 'b'],
    'pw_g': 14,
    'pw_a': 0,
    'hp': 200,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'bonus': ['a', 7, 'g'],
    'min': 150,
    'gas': 150,
    'supply': 3
    }
infestor = {
    'attributes': ['a', 'p', 'b'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 90,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'n',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 150,
    'supply': 2
    }
swarm_host = {
    'attributes': ['a', 'b'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 160,
    'sh': 0,
    'ar':1,
    'type': 'g',
    'target': 'n',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 75,
    'supply': 3
    }
ultralisk = {
    'attributes': ['a', 'b', 'massive'],
    'pw_g': 57.38,
    'pw_a': 0,
    'hp': 500,
    'sh': 0,
    'ar':2,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 300,
    'gas': 200,
    'supply': 6
    }
broodling = {
    'attributes': ['l', 'b'],
    'pw_g': 8.7,
    'pw_a': 0,
    'hp': 30,
    'sh': 0,
    'ar':0,
    'type': 'g',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 0,
    'gas': 0,
    'supply': 0
    }
changeling = {
    'attributes': ['l', 'b'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 5,
    'sh': 0,
    'ar': 0,
    'type': 'g',
    'target': 'n',
    'det': 'n',
    'inv': 'y',
    'min': 0,
    'gas': 0,
    'supply': 0
    }
overlord  = {
    'attributes': ['a', 'b'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 200,
    'sh': 0,
    'ar':0,
    'type': 'a',
    'target': 'n',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 0,
    'supply': 0
    }
overseer = {
    'attributes': ['a', 'b', 'd'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 200,
    'sh': 0,
    'ar':1,
    'type': 'a',
    'target': 'n',
    'det': 'y',
    'inv': 'n',
    'min': 150,
    'gas': 50,
    'supply': 0
    }
mutalisk = {
    'attributes': ['l', 'b'],
    'pw_g': 10,
    'pw_a': 10,
    'hp': 120,
    'sh': 0,
    'ar':0,
    'type': 'a',
    'target': 'ga',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 100,
    'supply': 2
    }
corruptor = {
    'attributes': ['a', 'b'],
    'pw_g': 0,
    'pw_a': 10.29,
    'hp': 200,
    'sh': 0,
    'ar':2,
    'type': 'a',
    'target': 'a',
    'det': 'n',
    'inv': 'n',
    'bonus': ['massive', 4.4, 'a'],
    'min': 150,
    'gas': 100,
    'supply': 2
    }
brood_lord = {
    'attributes': ['a', 'b', 'massive'],
    'pw_g': 11.2,
    'pw_a': 0,
    'hp': 225,
    'sh': 0,
    'ar':1,
    'type': 'a',
    'target': 'g',
    'det': 'n',
    'inv': 'n',
    'min': 300,
    'gas': 250,
    'supply': 4
    }
viper  = {
    'attributes': ['a', 'b', 'p'],
    'pw_g': 0,
    'pw_a': 0,
    'hp': 150,
    'sh': 1,
    'ar':0,
    'type': 'a',
    'target': 'n',
    'det': 'n',
    'inv': 'n',
    'min': 100,
    'gas': 200,
    'supply': 3
    }