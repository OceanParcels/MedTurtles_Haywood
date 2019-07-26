from parcels import FieldSet, Field, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable
from datetime import timedelta as delta
from datetime import datetime as datetime
from glob import glob
import numpy as np
import xarray as xr

wstokes = False

ddir = '/Volumes/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
# ufiles = sorted(glob(ddir+'means/ORCA0083-N06_20[00-10]*d05U.nc'))
ufiles = sorted(glob(ddir+'means/ORCA0083-N06_200[1-2]*d05U.nc'))
vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
meshfile = glob(ddir+'domain/coordinates.nc')

nemofiles = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
             'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}
nemovariables = {'U': 'uo', 'V': 'vo'}
nemodimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
nemoindices = {'lon': range(3350, 3885), 'lat': range(1875, 2140)}
fieldset_nemo = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, indices=nemoindices)

if wstokes:
#     stokesfiles = sorted(glob('/Volumes/oceanparcels/input_data/WaveWatch3data/CFSR/ww3.*_uss.nc'))
    stokesfiles = sorted(glob('/Volumes/oceanparcels/input_data/WaveWatch3data/CFSR/ww3.200[1-2]*_uss.nc'))
    stokesdimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    stokesvariables = {'U': 'uuss', 'V': 'vuss'}
    fieldset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
    fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

    fieldset = FieldSet(U=fieldset_nemo.U+fieldset_stokes.U, V=fieldset_nemo.V+fieldset_stokes.V)
    fU = fieldset.U[0]
else:
    fieldset = fieldset_nemo
    fU = fieldset.U

fieldset.computeTimeChunk(fU.grid.time[-1], -1)

startlats = [38.874, 38.216, 37.475, 37.453, 36.587, 36.537, 35.796, 35.653, 35.023, 37.005, 36.549, 36.442, 36.05, 36.038, 36.292, 36.574, 36.497, 35.878, 35.802, 36.052, 35.609, 35.354, 35.313, 34.898, 31.393, 31.494, 31.494, 37.658, 37.075, 35.86588, 35.51244, 35.797315, 35.231]
startlons = [20.3, 21.073, 20.991, 21.416, 22.182, 22.717, 23.956, 24.511, 24.429, 26.819, 28.454, 28.916, 29.1, 30.219, 30.785, 31.002, 31.315, 32.337, 32.962, 34.14, 33.326, 32.621, 32.366, 32, 33.582, 16.727, 16.43, 15.92, 13.277, 12.870452, 12.588662, 11.034225, 11.487]
nparticles = [5000, 5000, 124400, 68500, 5000, 19700, 9400, 32400, 5100, 6000, 19350, 22200, 8100, 26400, 5950, 44350, 16000, 8900, 41000, 9350, 12300, 7200, 10000, 14000, 6700, 10400, 33900, 1000, 1000, 1000, 1000, 1000, 1000]

import scipy.stats as stats
allstartlon = []
allstartlat = []
for s in range(len(startlats)):
    for i in range(int(nparticles[s]/100)):
        allstartlon.append(startlons[s] + np.random.uniform(-10, 10) / 1.852 / 60. / np.cos(startlats[s] * np.pi / 180))
        allstartlat.append(startlats[s] + np.random.uniform(-10, 10) / 1.852 / 60.)

def DeleteParticle(particle, fieldset, time):
    particle.delete()

for y in [2001]: # range(2001, 2006):
    if wstokes:
        fname = "medturtles_%d_wstokes.nc" % y
    else:
        fname = "medturtles_%d.nc" % y
    
    allstarttime = []
    for i in range(len(allstartlon)):
        ddays = int(stats.truncnorm(-30./20, 30./20, loc=0, scale=20).rvs(1))
        allstarttime.append(datetime(y, 8, 31, 0, 0, 0) + delta(days=ddays))
    
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=allstartlon, lat=allstartlat,
                       time=allstarttime)
    outfile = pset.ParticleFile(name=fname, outputdt=delta(days=1))

    pset.execute(AdvectionRK4, dt=delta(hours=1), endtime=datetime(y+1, 2, 1, 0, 0, 0),
                 output_file=outfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
    outfile.close()
