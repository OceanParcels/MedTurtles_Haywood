from parcels import FieldSet, Field, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable, VectorField
from datetime import timedelta as delta
from datetime import datetime as datetime
from glob import glob
import numpy as np
import xarray as xr
import scipy.stats as stats

wstokes = False
unbeaching = True
model = 'cmems'

if model == 'nemo':
    ddir = '/Volumes/oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/'
    ufiles = sorted(glob(ddir+'means/ORCA0083-N06_200[1-2]*d05U.nc'))
    vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
    meshfile = glob(ddir+'domain/coordinates.nc')

    nemofiles = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
                 'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}
    nemovariables = {'U': 'uo', 'V': 'vo'}
    nemodimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
    nemoindices = {'lon': range(3350, 3885), 'lat': range(1875, 2140)}
    fieldset_currents = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, indices=nemoindices)
elif model == 'cmems':
    ddir = '/Volumes/oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/'
    cmemsfiles = sorted(glob(ddir+'mercatorglorys12v1_gl12_mean_201[7-8]*.nc'))
    cmemsvariables = {'U': 'uo', 'V': 'vo'}
    cmemsdimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    cmemsindices = {'lon': range(2070, 2600), 'lat': range(1320, 1520)}
    fieldset_currents = FieldSet.from_netcdf(cmemsfiles, cmemsvariables, cmemsdimensions, indices=cmemsindices)

if wstokes:
    stokesfiles = sorted(glob('/Volumes/oceanparcels/input_data/WaveWatch3data/CFSR/WW3.201*_uss.nc'))
    stokesdimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    stokesvariables = {'U': 'uuss', 'V': 'vuss'}
    fieldset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
    fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

    fieldset = FieldSet(U=fieldset_currents.U+fieldset_stokes.U, V=fieldset_currents.V+fieldset_stokes.V)
    fU = fieldset.U[0]
else:
    fieldset = fieldset_currents
    fU = fieldset.U

if unbeaching:
    if model == 'nemo':
        variables = {'U_unbeach': 'unBeachU', 'V_unbeach': 'unBeachV'}
        dimensions = {'lon': 'glamf', 'lat': 'gphif'}
        fieldsetUnBeach = FieldSet.from_nemo(glob(ddir+'domain/ORCA0083-N06_unbeaching_vel.nc'), variables, dimensions, tracer_interp_method='cgrid_velocity', indices=nemoindices)
    elif model == 'cmems':
        variables = {'U_unbeach': 'unBeachU', 'V_unbeach': 'unBeachV'}
        dimensions = {'lon': 'longitude', 'lat': 'latitude'}
        fieldsetUnBeach = FieldSet.from_netcdf(glob(ddir+'mercatorglorys12v1_gl12_unbeaching_vel.nc'), variables, dimensions, indices=cmemsindices)
        fieldsetUnBeach.U_unbeach.units = fieldset.U.units
        fieldsetUnBeach.V_unbeach.units = fieldset.V.units

    fieldset.add_field(fieldsetUnBeach.U_unbeach)
    fieldset.add_field(fieldsetUnBeach.V_unbeach)

    UVunbeach = VectorField('UVunbeach', fieldset.U_unbeach, fieldset.V_unbeach)
    fieldset.add_vector_field(UVunbeach)

startlats = [35.313, 35.354, 34.954, 35.609, 35.178, 35.239, 35.59, 35.609, 35.248, 35.668, 35.849, 37.475, 37.324, 35.653, 36.537, 37.333, 35.796, 37.488, 37.005, 36.587, 35.023, 38.874, 37.866, 36.867, 39.227, 33.343, 31.497, 31.529, 31.465, 31.489, 32.099, 32.439, 31.491, 31.856, 32.325, 31.749, 33.046, 32.022, 35.797, 36.574, 35.802, 36.549, 36.044, 36.497, 35.878, 36.442, 36.052, 36.05, 36.433, 35.988, 36.292, 36.211]
startlons = [32.366, 32.621, 31.975, 33.326, 34.317, 34.231, 32.928, 33.528, 34.454, 33.69, 34.202, 20.991, 21.356, 24.511, 22.717, 21.341, 23.956, 21.412, 26.819, 22.182, 24.429, 20.3, 20.543, 21.35, 20.308, 34.925, 16.414, 16.164, 19.692, 16.371, 15.671, 23.505, 16.383, 15.725, 15.606, 15.79, 21.299, 15.671, 11.034, 31.002, 32.962, 28.454, 30.187, 31.315, 32.337, 28.916, 34.14, 29.1, 23.689, 30.068, 30.785, 34.32]
nparticles = [37200, 8900, 10900, 5900, 4800, 4500, 4100, 3800, 3700, 3500, 2900, 121800, 78100, 35000, 19700, 10000, 9400, 6400, 6000, 5300, 5100, 5000, 2900, 2200, 2000, 5500, 12200, 11900, 10400, 7300, 3400, 2600, 2500, 2300, 2200, 2000, 1700, 1600, 1100, 46600, 42200, 25000, 18400, 13900, 10900, 10100, 9900, 9300, 8800, 6700, 5100, 2500]

np.random.seed(1234)
allstartlon = []
allstartlat = []
for s in range(len(startlats)):
    for i in range(int(nparticles[s]/100)):
        allstartlon.append(startlons[s] + np.random.uniform(-10, 10) / 1.852 / 60. / np.cos(startlats[s] * np.pi / 180))
        allstartlat.append(startlats[s] + np.random.uniform(-10, 10) / 1.852 / 60.)

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def UnBeaching(particle, fieldset, time):
    (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    spd = math.sqrt(math.pow(u*1852*60*math.cos(particle.lat*3.1415/180.), 2) + math.pow(v*1852*60, 2))
    if spd < 1e-3:
        (ub, vb) = fieldset.UVunbeach[time, particle.depth, particle.lat, particle.lon]
        particle.lon += ub * particle.dt
        particle.lat += vb * particle.dt

for y in [2017]:
    ws = '_wstokes' if wstokes else ''
    ub = '_unbeach' if unbeaching else ''
    fname = "medturtles_%s_%d%s%s.nc" % (model, y, ws, ub)

    allstarttime = []
    for i in range(len(allstartlon)):
        ddays = int(stats.truncnorm(-30./20, 30./20, loc=0, scale=20).rvs(1))
        allstarttime.append(datetime(y, 8, 31, 0, 0, 0) + delta(days=ddays))

    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=allstartlon, lat=allstartlat,
                       time=allstarttime)
    outfile = pset.ParticleFile(name=fname, outputdt=delta(days=1))

    if unbeaching == True:
        kernels = pset.Kernel(AdvectionRK4) + UnBeaching
    elif unbeaching is False:
        kernels = pset.Kernel(AdvectionRK4)

    pset.execute(kernels, dt=delta(hours=1), endtime=datetime(y+1, 7, 31, 0, 0, 0), output_file=outfile)
    outfile.close()
