from parcels import FieldSet, Field, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable, VectorField
from datetime import timedelta as delta
from datetime import datetime as datetime
from glob import glob
import numpy as np
import xarray as xr
import scipy.stats as stats

ddir = '/data/'
def run_medturtles(wstokes, unbeaching, years, model='cmems'):

    yrnum = years[0]-2010
    if model == 'nemo':
        ufiles = sorted(glob(ddir+'oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/means/ORCA0083-N06_200[%d-%d]*d05U.nc' %(yrnum, yrnum+1)))
        vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
        meshfile = glob(ddir+'oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/domain/coordinates.nc')
        nemofiles = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
                     'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}
        nemovariables = {'U': 'uo', 'V': 'vo'}
        nemodimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
        nemoindices = {'lon': range(3350, 3885), 'lat': range(1875, 2140)}
        fset_currents = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, indices=nemoindices)
    elif model == 'cmems':
        cmemsfiles = sorted(glob(ddir+'oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/mercatorglorys12v1_gl12_mean_201[%d-%d]*.nc' %(yrnum, yrnum+1)))
        cmemsvariables = {'U': 'uo', 'V': 'vo'}
        cmemsdimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
        cmemsindices = {'lon': range(2070, 2600), 'lat': range(1320, 1520)}
        fset_currents = FieldSet.from_netcdf(cmemsfiles, cmemsvariables, cmemsdimensions, indices=cmemsindices)

    if wstokes:
        stokesfiles = sorted(glob(ddir+'oceanparcels/input_data/WaveWatch3data/ECMWF/WW3-GLOB-30M_201[%d-%d]*_uss.nc' %(yrnum, yrnum+1)))
        stokesdimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
        stokesvariables = {'U': 'uuss', 'V': 'vuss'}
        fset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
        fset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

    if unbeaching:
        if model == 'nemo':
            variables = {'U': 'unBeachU', 'V': 'unBeachV'}
            dimensions = {'lon': 'glamf', 'lat': 'gphif'}
            fset_unbeach = FieldSet.from_nemo(glob(ddir+'oceanparcels/input_data/NEMO-MEDUSA/ORCA0083-N006/domain/ORCA0083-N06_unbeaching_vel.nc'), variables, dimensions, tracer_interp_method='cgrid_velocity', indices=nemoindices)
        elif model == 'cmems':
            variables = {'U': 'unBeachU', 'V': 'unBeachV'}
            dimensions = {'lon': 'longitude', 'lat': 'latitude'}
            fset_unbeach = FieldSet.from_netcdf(glob(ddir+'oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/mercatorglorys12v1_gl12_unbeaching_vel.nc'), variables, dimensions, indices=cmemsindices)

    if wstokes:
        if unbeaching:
            fieldset = FieldSet(U=fset_currents.U+fset_stokes.U+fset_unbeach.U, 
                                V=fset_currents.V+fset_stokes.V+fset_unbeach.V)
        else:
            fieldset = FieldSet(U=fset_currents.U+fset_stokes.U, V=fset_currents.V+fset_stokes.V)
    else:
        if unbeaching:
            fieldset = FieldSet(U=fset_currents.U+fsetUnBeach.U, 
                                V=fset_currents.V+fsetUnBeach.V)
        else:
            fieldset = fset_currents

    startlats = [35.313, 34.954, 35.354, 35.609, 35.178, 35.239, 35.59, 35.609, 35.248, 35.668, 35.849, 37.475, 37.324, 35.653, 36.537, 37.333, 35.796, 37.488, 37.005, 36.587, 35.023, 38.874, 37.866, 36.867, 39.227, 33.343, 31.497, 31.529, 31.465, 31.489, 32.099, 32.439, 31.491, 31.856, 32.325, 31.749, 33.046, 32.022, 35.797315, 36.574, 35.802, 36.549, 36.044, 36.497, 35.878, 36.442, 36.052, 36.05, 36.433, 35.988, 36.292, 36.211]
    startlons = [32.366, 31.975, 32.621, 33.326, 34.317, 34.231, 32.928, 33.528, 34.454, 33.69, 34.202, 20.991, 21.356, 24.511, 22.717, 21.341, 23.956, 21.412, 26.819, 22.182, 24.429, 20.3, 20.543, 21.35, 20.308, 34.925, 16.414, 16.164, 19.692, 16.371, 15.671, 23.505, 16.383, 15.725, 15.606, 15.79, 21.299, 15.671, 11.034225, 31.002, 32.962, 28.454, 30.187, 31.315, 32.337, 28.916, 34.14, 29.1, 23.689, 30.068, 30.785, 34.32]
    nparticles = [372, 109, 89, 59, 48, 45, 41, 38, 37, 35, 29, 1218, 781, 350, 197, 100, 94, 64, 60, 53, 51, 50, 29, 22, 20, 55, 122, 119, 104, 73, 34, 26, 25, 23, 22, 20, 17, 16, 11, 466, 422, 250, 184, 139, 109, 101, 99, 93, 88, 67, 51, 25]

    nppersite = 1.
    np.random.seed(1234)
    allstartlon = []
    allstartlat = []
    allsitenumbers = []
    for s in range(len(startlats)):
        for i in range(int(nparticles[s]*nppersite)):
            allstartlon.append(startlons[s] + np.random.uniform(-10, 10) / 1.852 / 60. / np.cos(startlats[s] * np.pi / 180))
            allstartlat.append(startlats[s] + np.random.uniform(-10, 10) / 1.852 / 60.)
            allsitenumbers.append(s+1)

    def SampleSpeed(particle, fieldset, time):
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        particle.u = u*1852*60*math.cos(particle.lat*3.1415/180.)
        particle.v = v*1852*60

    for y in years:
        ws = '_wstokes' if wstokes else ''
        ub = '_unbeach' if unbeaching else '_beach'
        fname = "medturtles_%s_%d%s%s.nc" % (model, y, ws, ub)

        allstarttime = []
        for i in range(len(allstartlon)):
            ddays = int(stats.truncnorm(-30./20, 30./20, loc=0, scale=20).rvs(1))
            allstarttime.append(datetime(y, 8, 31, 0, 0, 0) + delta(days=ddays))

        class TurtleParticle(JITParticle):
            u = Variable('u', dtype=np.float32)
            v = Variable('v', dtype=np.float32)
            sitenumber = Variable('sitenumber', dtype=np.int32, to_write='once')

        pset = ParticleSet(fieldset=fieldset, pclass=TurtleParticle, lon=allstartlon, lat=allstartlat,
                           time=allstarttime, sitenumber=allsitenumbers)
        outfile = pset.ParticleFile(name=fname, outputdt=delta(days=1))

        kernels = pset.Kernel(AdvectionRK4) + SampleSpeed
        pset.execute(kernels, dt=delta(hours=1), endtime=datetime(y+1, 7, 31, 0, 0, 0), output_file=outfile)
        outfile.close()


for yr in np.arange(2016, 2018):
    for unbeaching in [True, False]:
        for wstokes in [False, True]:
            run_medturtles(wstokes, unbeaching, [yr])
