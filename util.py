#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, get_moon, get_sun, get_body
from astropy.coordinates.name_resolve import NameResolveError
import astropy.table

from collections import OrderedDict

import ephem
import pyorbital.orbital

import functools
from matplotlib.mlab import contiguous_regions  # deprecated (moved to matplotlib.cbook)
import matplotlib.collections as mcoll

import sys, os



# -------------------------- Start settings section -------------------------- #

# fonts sizes
fonts = { 'skymap': 10,
          'const_name': 10,
          'fov': 10,
          'legend': 10,
          'title': 24,
          'caption': 12 }

# markers sizes
markers = { 'moon': 25,
            'sun': 25,
            'iss': 10,
            'planet': 10 }

# 'light' or 'dark'
style = 'dark'

# relative to the figure size (0-1) and not including axes ticklabels
plot_size = 0.95

dpi = 80

# --------------------------- End settings section --------------------------- #



if style == 'light':
    colors = { 'fov_outer': 'orange' }
    plt.style.use('fivethirtyeight')
elif style == 'dark':
    colors = { 'fov_outer': 'lightseagreen' }
    plt.style.use('dark_background')



def download_stars(constellations_dict):
    """
    Form astropy.Table with constellations, their stars and (RA,dec) coordinates

    input:
        dictionary in format {'constellation_3-letter_code': [list_of_stars'_names]}

    returns:
        Table instance
    """

    print('downloading {} constellations...'.format(len(constellations_dict.keys())))

    stars_table = astropy.table.Table( names=['Constellation', 'Star', 'RA', 'dec'],
                                       meta={'name': "constellations' stars"},
                                       dtype=['object', 'object', 'float', 'float'] )

    # fill this table with data
    skipped_stars = 0
    for name,stars in constellations_dict.items():
        unique_stars = list(set(stars))

        for letter in unique_stars:
            search_request = letter + ' ' + name
            print("\r{}".format(search_request.ljust(80)), end='')
            try:
                star = SkyCoord.from_name(search_request)
                stars_table.add_row([name, letter, star.ra, star.dec])
            except NameResolveError:
                print("\rWarning: {} not found!".format(search_request).ljust(80))
                skipped_stars += 1
                continue

    print( "\r{} stars were downloaded, {} were skipped"
           .format(len(stars_table['Star']), skipped_stars).ljust(80) )
    if skipped_stars > 0:
        print("You will not be able to plot constellations. Try to re-download")

    return stars_table



def extract_constellations(stars):
    """
    Extract individual constellations from a one table and put them into separate
    tables stored in a list (sorted by constellations' names)

    returns:
        list with constellations' astropy.Table tables
    """

    constellations = []

    # find unique constellations in db
    for i,constellation in enumerate(sorted(np.unique(stars['Constellation']))):
        constellations.append( astropy.table.Table( names=['Constellation', 'Star', 'RA', 'dec'],
                                                    dtype=['object', 'object', 'float', 'float'] ) )
        for star in stars:
            if star['Constellation'] == constellation:
                constellations[i].add_row(star)

    print("extracted {} stars from {} constellations".format(len(stars), len(constellations)))

    return constellations



def extract_forms(constellations_dict, stars_table):
    """
    Define indexes of stars in the database table that corresponds to stars in constellations' forms

    returns:
        astropy.Table with 3 columns: constellation name,
                                      path (with stars' letters) (for human-reading),
                                      path (with stars' indexes) (for machine-reading)
    """

    constellations_forms = astropy.table.Table( names=['constellation', 'path', 'idxs'],
                                                dtype=['object', 'object', 'object'] )

    for (name,path),constellation in zip( constellations_dict.items(),
                                          extract_constellations(stars_table) ):
        idxs = ''
        for letter in path:
            idxs = idxs + str(list(constellation['Star']).index(letter)) + ' '

        path_for_table = ''
        for elem in path:
            path_for_table = path_for_table + elem + ' '

        constellations_forms.add_row([name, path_for_table, idxs])

    return constellations_forms



def get_data(stars_db_name, constellations_db_name):
    """
    Get necessary data from database' files and do some checks
    """

    # read stars table database from the file to Table object
    if os.path.isfile(stars_db_name):
        stars = astropy.table.Table.read(stars_db_name)
    else:
        print("{} not found. Run 'dl_constellations.py'".format(stars_db_name))
        sys.exit()

    # generate constellations forms if there is no one
    if not os.path.isfile(constellations_db_name):
        from dl_constellations import constellations as constellations_dict
        constellations_forms = extract_forms(constellations_dict, stars)
        constellations_forms.write(constellations_db_name, format='ascii.html', overwrite=True)
    else:
        constellations_forms = astropy.table.Table.read(constellations_db_name)

    # exctract each constellation
    constellations = extract_constellations(stars)

    return constellations,constellations_forms



def generate_polycollection(ax, x, y1, y2):
    """
    Accessory function that creates new PolyCollection class instance representing
    new correct field-of-view. We use its output to assign new *path* to the old
    one instance. Algorithm has been taken from the matplotlib' sources for the
    fill_between() function and used because there is no easy way to update polygon
    with new data

    input:
        same as for the fill_between() function

    returns:
        matplotlib.collections.PolyCollection instance
    """

    # i.e. where = outer_circle >= fov_circle
    where = y2>=y1
    kwargs = { 'facecolor': colors['fov_outer'],
               'alpha': 0.25 }

    # Convert the arrays so we can work with them
    x = np.ma.masked_invalid(ax.convert_xunits(x))
    y1 = np.ma.masked_invalid(ax.convert_yunits(y1))
    y2 = np.ma.masked_invalid(ax.convert_yunits(y2))

    where = where & ~functools.reduce( np.logical_or,
                                       map(np.ma.getmask, [x, y1, y2]) )

    x, y1, y2 = np.broadcast_arrays(np.atleast_1d(x), y1, y2)

    polys = []
    for ind0, ind1 in contiguous_regions(where):
        xslice = x[ind0:ind1]
        y1slice = y1[ind0:ind1]
        y2slice = y2[ind0:ind1]

        if not len(xslice):
            continue

        N = len(xslice)
        X = np.zeros((2 * N + 2, 2), float)

        # the purpose of the next two lines is for when y2 is a
        # scalar like 0 and we want the fill to go all the way
        # down to 0 even if none of the y1 sample points do
        start = xslice[0], y2slice[0]
        end = xslice[-1], y2slice[-1]

        X[0] = start
        X[N + 1] = end

        X[1:N + 1, 0] = xslice
        X[1:N + 1, 1] = y1slice
        X[N + 2:, 0] = xslice[::-1]
        X[N + 2:, 1] = y2slice[::-1]

        polys.append(X)

    return mcoll.PolyCollection(polys, **kwargs)



class StarChart():
    """
    Class completely representing sky map
    """


    def __init__(self):
        """
        At creation, we generate Figure itself, background Axes and plot static
        objects (constellations, labels)
        """

        self.fig,self.ax = self.generate_fig_ax(fontsize=fonts['skymap'])
        constellations,constellations_forms = get_data('stars.html', '.constellations.html')
        self.plot_constellations(constellations, constellations_forms)



    def get_fig_ax(self):
        """
        For using with external code
        """

        return self.fig,self.ax



    def prepare(self, obs_time, obs_loc, tle):
        """
        Plot remaining elements of the map based on current time, location and
        TLE (for ISS)

        Returns:
            Tuple of elements that need to be redrawn at each animation update
        """

        self.obs_loc = obs_loc
        self.tle = tle if tle else None

        self.plot_fov(obs_time, fontsize=10)
        self.plot_moon(obs_time)
        self.plot_sun(obs_time)
        self.plot_solarsystem(obs_time)
        self.plot_iss(obs_time, tle=self.tle)

        return self.fov_line + self.fov_polygon + self.fov_ticklabels +\
               self.SN_line + self.WE_line + self.iss_line



    def update(self, obs_time):
        """
        Call this function when you want to redraw moving elements of the map
        """

        self.update_fov(obs_time)
        self.update_iss(obs_time, self.tle)



    def generate_fig_ax(self, fontsize=10):
        """
        Form outer circle for ICRS coordinate system

        returns:
            matplotlib Figure and Axes instances
        """

        fig = plt.figure(dpi=dpi)
        fig.canvas.set_window_title('starmap-northern-animation')

        # docs quote: add an axes at position [left, bottom, width, height] where
        # all quantities are in fractions of figure width and height
        ax = fig.add_axes([-0.185, 0.025, plot_size, plot_size], polar=True)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # anti-clockwise

        ax.set_rlim(90, -45)  # same as set_ylim()
        ax.set_yticks(np.arange(-45, 90+0.1, 15))

        ax.grid(True)
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle(':')
            line.set_linewidth(0.5)

        for yticklabel in ax.get_yticklabels():
            yticklabel.set_visible(False)
        for xticklabel in ax.get_xticklabels():
            xticklabel.set_visible(False)

        print("sky map is prepared")

        return fig,ax



    def plot_constellations(self, constellations, constellations_forms):
        """
        Plotting constellations on axes with ICRS coordinate system
        """

        for constellation,constellation_form in zip(constellations,constellations_forms['idxs']):
            constellation_form = [int(s) for s in constellation_form.split()]

            # put stars on plot and connect them by pairs
            for i,iplus1 in zip(constellation_form[:-1], constellation_form[1:]):
                self.ax.plot( np.radians([ constellation['RA'][i], constellation['RA'][iplus1] ]),
                              [ -constellation['dec'][i]+45, -constellation['dec'][iplus1]+45 ],
                              color='skyblue', marker=None, linewidth=2.0 )

            # put name of constellation beside its first star
            self.ax.text( np.radians(constellation['RA'][0]),
                     -constellation['dec'][0]+45,
                     "{}".format(constellation['Constellation'][0]),
                     fontsize=fonts['const_name'], weight='bold' )

            # print progress
            print("\r{} plotted".format(constellation['Constellation'][0]).ljust(80), end='')

        print('\rconstellations are plotted'.ljust(80))



    def plot_fov(self, obs_time, fontsize=10):
        """
        Plot initial position of the field-of-view. Here we also creating matplotlib'
        objects that needs to be redrawn at each animation iteration. We immediately
        assign them to the class instance instead of returning to the caller because
        otherwise we should unpack all of them and hard-code indexes of elements in
        the resulted tuple
        """

        #
        # plotting field-of-view circle
        #
        self.fov_az = np.arange(0, 360+0.1, 5)
        self.fov_alt = np.zeros(len(self.fov_az))
        fov = SkyCoord( self.fov_az, self.fov_alt, unit='deg', frame='altaz', obstime=obs_time,
                        location=self.obs_loc ).transform_to('icrs')

        self.fov_line = tuple( self.ax.plot( fov.ra.radian, -fov.dec.value+45, '-',
                                   linewidth=0.5, color=colors['fov_outer'] ) )


        #
        # fill the area that we cannot observe now
        #
        shared_ax = fov.ra.radian
        fov_circle = -fov.dec.value+45
        self.outer_circle = len(fov_circle) * [-(-45)+45]

        self.fov_polygon = ( self.ax.fill_between(
            shared_ax, fov_circle, self.outer_circle, where=self.outer_circle>=fov_circle,
            facecolor=colors['fov_outer'], alpha=0.25 ),
        )


        #
        # putting on plot ticks of circle axis (axis of azimuth) in the same way as outer circle
        #
        self.fov_ticks_az = np.arange(0, 270+0.1, 90)
        self.fov_ticks_alt = np.zeros(len(self.fov_ticks_az))
        fov_ticks = SkyCoord( self.fov_ticks_az, self.fov_ticks_alt, unit='deg', frame='altaz',
                              obstime=obs_time, location=self.obs_loc ).transform_to('icrs')
        cardinal_directions = ['N', 'E', 'S', 'W']  # anti-clockwise

        self.fov_ticklabels = ()
        for letter,tick_coord in zip(cardinal_directions,fov_ticks):
            self.fov_ticklabels += (
                self.ax.text( tick_coord.ra.radian, -tick_coord.dec.value+45, letter,
                              fontsize=fontsize*2, fontname="Apple Chancery", fontweight='bold' ), )

        #
        # plot straight axis - from South to North - of field of view circle (similarly)
        #
        self.SN_ax_alt = [0, 0]
        self.SN_ax_az = [0, 180]
        SN_ax = SkyCoord( self.SN_ax_az, self.SN_ax_alt, unit='deg', frame='altaz',
                          obstime=obs_time, location=self.obs_loc ).transform_to('icrs')

        self.SN_line = tuple( self.ax.plot(
            SN_ax.ra.radian, -SN_ax.dec.value+45, '-', linewidth=0.5, color=colors['fov_outer'])
        )

        #
        # plot curved axis - from West to East - of field of view circle (similarly)
        #
        self.WE_ax_alt = [alt for alt in np.arange(0, 90+0.1, 1)]
        self.WE_ax_az = len(self.WE_ax_alt)*[90] + (len(self.WE_ax_alt)-1)*[270]
        self.WE_ax_alt = self.WE_ax_alt + self.WE_ax_alt[:-1][::-1]
        WE_ax = SkyCoord( self.WE_ax_az, self.WE_ax_alt, unit='deg', frame='altaz',
                          obstime=obs_time, location=self.obs_loc ).transform_to('icrs')

        self.WE_line = tuple( self.ax.plot(
            WE_ax.ra.radian, -WE_ax.dec.value+45, '-', linewidth=0.5, color=colors['fov_outer'])
        )

        print("field-of-view is plotted")



    def update_fov(self, obs_time):
        """
        Function similarly to the plot_fov() but handles the redrawing of changed items.
        It gets previously created matplotlib instances to reuse them (increase performance)
        """

        # converting this coordinates to (RA,dec) format for plotting them onto plot
        fov = SkyCoord( self.fov_az, self.fov_alt, unit='deg', frame='altaz', obstime=obs_time,
                        location=self.obs_loc ).transform_to('icrs')
        # plotting field-of-view circle (set new updated data)
        self.fov_line[0].set_data(fov.ra.radian, -fov.dec.value+45)


        # fill the area that we cannot observe now
        shared_ax = fov.ra.radian
        fov_circle = -fov.dec.value+45
        collection = generate_polycollection(self.ax, shared_ax, fov_circle, self.outer_circle)
        self.fov_polygon[0].get_paths()[0] = collection.get_paths()[0]


        # putting on plot ticks of circle axis (axis of azimuth) in the same way as outer circle
        fov_ticks = SkyCoord( self.fov_ticks_az, self.fov_ticks_alt, unit='deg', frame='altaz',
                              obstime=obs_time, location=self.obs_loc ).transform_to('icrs')
        for i,tick_coord in enumerate(fov_ticks):
            self.fov_ticklabels[i].set_position(( fov_ticks[i].ra.radian, -fov_ticks[i].dec.value+45 ))


        # plot straight axis (from South to North) of field of view circle (similarly)
        SN_ax = SkyCoord( self.SN_ax_az, self.SN_ax_alt, unit='deg', frame='altaz',
                          obstime=obs_time, location=self.obs_loc ).transform_to('icrs')
        self.SN_line[0].set_data( SN_ax.ra.radian, -SN_ax.dec.value+45 )


        # plot curved axis (from West to East) of field of view circle (similarly)
        WE_ax = SkyCoord( self.WE_ax_az, self.WE_ax_alt, unit='deg', frame='altaz',
                          obstime=obs_time, location=self.obs_loc ).transform_to('icrs')
        self.WE_line[0].set_data( WE_ax.ra.radian, -WE_ax.dec.value+45 )



    def plot_iss(self, obs_time, tle=None):
        """
        Initial placing of the International Space Station on the plot. The function
        uses TLE list if it was given or tries to retrieve it via PyOrbital package.
        We do not use it anywhere after first inclusion
        """

        if tle is None:
            print("request ISS' two-line elements via PyOrbital...")
            self.pyorbital_orbital = pyorbital.orbital.Orbital("ISS (ZARYA)")
            iss = self.pyorbital_orbital.get_observer_look( obs_time.value,
                                                            self.obs_loc.lon.value,
                                                            self.obs_loc.lat.value,
                                                            self.obs_loc.height.value )
            iss_coord = SkyCoord( iss[0], iss[1], unit='deg', frame='altaz',
                                  obstime=obs_time, location=self.obs_loc ).transform_to('icrs')
        else:
            self.iss = ephem.readtle('ISS', tle[0], tle[1])
            self.ephem_observer = ephem.Observer()
            self.ephem_observer.lat = str(self.obs_loc.lat.value)
            self.ephem_observer.lon = str(self.obs_loc.lon.value)
            self.ephem_observer.date = obs_time.value
            self.iss.compute(self.ephem_observer)
            iss_coord = SkyCoord( self.iss.az, self.iss.alt, unit='rad', frame='altaz',
                                  obstime=obs_time, location=self.obs_loc ).transform_to('icrs')

        self.iss_line = tuple( self.ax.plot(
            [iss_coord.ra.radian], [-iss_coord.dec.value+45], label='ISS', linestyle='',
            color='green', marker='$⋈$', markersize=markers['iss'] )
        )

        print("ISS is plotted")



    def update_iss(self, obs_time, tle):
        """
        Update position of the ISS on the plot (use the same TLE) whether it was
        created by PyOrbital or PyEphem
        """

        if tle is None:
            iss = self.pyorbital_orbital.get_observer_look( obs_time.value,
                                                            self.obs_loc.lon.value,
                                                            self.obs_loc.lat.value,
                                                            self.obs_loc.height.value )
            iss_coord = SkyCoord( iss[0], iss[1], unit='deg', frame='altaz',
                                  obstime=obs_time, location=self.obs_loc ).transform_to('icrs')
        else:
            self.ephem_observer.date = obs_time.value
            self.iss.compute(self.ephem_observer)
            iss_coord = SkyCoord( self.iss.az, self.iss.alt, unit='rad', frame='altaz',
                                  obstime=obs_time, location=self.obs_loc ).transform_to('icrs')

        self.iss_line[0].set_data( iss_coord.ra.radian, -iss_coord.dec.value+45 )



    def plot_moon(self, obs_time):
        """
        Put Moon on a given Axes instance
        """

        moon = get_moon(obs_time, location=self.obs_loc)
        moon = SkyCoord(moon.ra, moon.dec, frame='gcrs').transform_to('icrs')

        self.ax.plot( [moon.ra.radian], [-moon.dec.value+45], label='Moon', linestyle='',
                      color='indianred', marker='$☽$', markersize=markers['moon'] )

        print("Moon is plotted")



    def plot_sun(self, obs_time):
        """
        Put Sun on a given Axes instance
        """

        sun = get_sun(obs_time)
        sun = SkyCoord(sun.ra, sun.dec, frame='gcrs').transform_to('icrs')

        self.ax.plot( [sun.ra.radian], [-sun.dec.value+45], label='Sun', linestyle='',
                      color='yellow', marker='$☀︎$', markersize=markers['sun'] )

        print("Sun is plotted")



    def plot_solarsystem(self, obs_time):
        """
        Put planets of Solar System on a given Axes instance. Designation of the
        each planet is its roman symbol
        """

        # for Uranus must be another symbol, but at Unicode U+2645, which renders as ♅ (Wiki)
        planets = OrderedDict([ ('Mercury', '☿'),
                                ('Venus', '♀'),
                                ('Mars', '♂'),
                                ('Jupiter', '♃'),
                                ('Saturn', '♄'),
                                ('Uranus', '♅'),
                                ('Neptune', '♆') ])

        planets_coords = [get_body(planet, obs_time, location=self.obs_loc) for planet in planets.keys()]
        for coords,(name,symbol) in zip(planets_coords,planets.items()):
            planet = SkyCoord(coords.ra, coords.dec, frame='gcrs').transform_to('icrs')
            self.ax.plot( [planet.ra.radian], [-planet.dec.value+45], label=name, linestyle='',
                          color='violet', marker='$'+symbol+'$', markersize=markers['planet'] )

        print("planets are plotted")



#
# test run (only Figure, Axes and constellations)
#
if __name__ == '__main__':

    import matplotlib
    # matplotlib.use('Qt5Agg')
    matplotlib.rcParams['toolbar']='None'
    import matplotlib.pyplot as plt

    star_chart = StarChart()
    fig,ax = star_chart.get_fig_ax()

    plt.show()
