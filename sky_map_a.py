#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# TODO: different stars sizes dependending on their flux
# TODO: map where only FOV and center is a current 'polar star' analogue (zenith)
# TODO: case when not all stars were downloaded
# TODO: implement constellations as graphs
# TODO: ask for lateset matplotlib fix



import sys

import numpy as np

import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.rcParams['toolbar']='None'
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import astropy.time
from astropy.coordinates import EarthLocation
import astropy.units as u

from datetime import datetime, timedelta
import time

from util import StarChart, fonts, plot_size



# -------------------------- Start settings section -------------------------- #

#
# Set desired location here. Specify name (required for caption), choose online/offline
# flag and in the last case also set latitude, longitude and altitude in the class Loc
#
location_str = "Syktyvkar"
internet_geolocation = 1
if not internet_geolocation:
    class Loc:
        """
        Hard-coded location. You can use it instead of Internet geolocationing
        """
        latitude = 61.40
        longitude = 50.49
        altitude = 100
        address = location_str


#
# Leave custom_time=None or custom_time=0 to use current OS time or specify
# the custom time as datetime(year, month, day, hours, minutes, seconds, ms)
#
custom_time = None  # datetime(2018, 4, 29, 0, 19, 42, 00000)


#
# Choose what TLE to use (only one key at a time can have '1' value).
#   'internet':            use pyobject to retrieve TLE
#   'internet_by_url':     use custom URL to download TLE
#   'local':               specify list for TLE by yourself
# Set all keys to '0' values to skip ISS plotting
#
tle_type = { 'internet':            1,
             'internet_by_url':     0,
             'local':               0 }
tle_url = 'https://www.celestrak.com/NORAD/elements/stations.txt'
tle_local = [ '1 25544U 98067A   18116.84910257  .00002058  00000-0  38261-4 0  9992',
              '2 25544  51.6422 274.6376 0002768  23.1132 122.6984 15.54167281110534' ]


# animation_interval of calling of the updating handler, ms
animation_interval = 250  # ms

# multiplier of the time
time_scale = 10


benchmark = False

# --------------------------- End settings section --------------------------- #



#
# handle observation geolocation
#
location = None

if internet_geolocation:
    # request geolocation by string
    print("request geolocation via geopy...")
    from geopy.geocoders import Nominatim as geocoder
    location = geocoder().geocode(location_str, timeout=5)
else:
    location = Loc()

if location is not None:
    print("observation location: " + location.address)
    obs_loc = EarthLocation( lat=location.latitude*u.deg,
                             lon=location.longitude*u.deg,
                             height=location.altitude*u.m  )
else:
    print("NO LOCATION")  # we can't proceed without location
    sys.exit()


#
# handle observation time
#
obs_time = None

# Moscow time: 3*u.hour (or datetime.now()-datetime.utcnow())
utc_offset = (time.timezone if (time.localtime().tm_isdst==0) else time.altzone)/60/60*-1

if custom_time:
    from math import modf
    obs_time = astropy.time.Time( custom_time - timedelta( hours=modf(utc_offset)[1],
                                                           minutes=modf(utc_offset)[0] ) )
else:
    obs_time = astropy.time.Time( datetime.utcnow() )

if not obs_time:
    print("NO TIME")  # we can't proceed without time
    sys.exit()



#
# plot ISS (or just skip)
#
tle = None

if tle_type['internet']:
    pass
elif tle_type['internet_by_url']:
    import urllib.request
    print("request ISS' two-line elements from URL...")
    with urllib.request.urlopen(tle_url) as tle_file:
        tle = [line.decode('utf-8')[:-2] for line in tle_file][1:3]
elif tle_type['local']:
    tle = tle_local



#
# class representing sky map
#
star_chart = StarChart()
fig,ax = star_chart.get_fig_ax()
elements_to_animate = star_chart.prepare(obs_time, obs_loc, tle if tle else None)



#
# title, caption with location & time and legend
#
fig.suptitle( "Star map of the\nnorthern semisphere",
              fontsize=fonts['title'], fontname="Andale Mono",
              x=plot_size-0.15*plot_size,y=(plot_size/2)+(plot_size/6) )

# we need to pack Text instance into tuple for animation functionality
text_line = tuple([ ax.text( np.radians(0), -(-30)+45,
    "{0}, ( {1:.2f}, {2:.2f} )\n\
     {3}\n".format( location_str, obs_loc.lat, obs_loc.lon,
                    str(obs_time + (utc_offset*u.hour)) ),
    fontsize=fonts['caption'], fontname="Andale Mono",
    horizontalalignment='center', verticalalignment='center' ) ])

# position is relative to the axes size (0-1)
ax.legend( labelspacing=2, fontsize=fonts['legend'], handletextpad=2, borderpad=2,
           ncol=5, bbox_to_anchor=(1+0.05, 0.425) )



#
# animation and showing
#

# objects to redraw
anim_tuple = elements_to_animate + text_line

if benchmark:
    import time
    start = time.time()

def update(frame):
    """
    Handler function to update the plot - only changed items. Currently we redraw
    only Field-of-View, ISS (most speedy object) and time caption.
    """

    if benchmark:
        end = time.time()
        global start
        print(end-start)
        start = end

    global obs_time
    obs_time += animation_interval * time_scale * u.ms

    star_chart.update(obs_time)
    text_line[0].set_text( "{0}, ( {1:.2f}, {2:.2f} )\n\
                            {3}\n".format( location_str, obs_loc.lat, obs_loc.lon,
                            str(obs_time + (utc_offset*u.hour))[:19] ) )

    # objects to redraw
    return anim_tuple

ani = animation.FuncAnimation( fig, update, init_func=lambda: anim_tuple,
                               interval=animation_interval, blit=True )

plt.show()
