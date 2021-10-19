# Script to generalise api request for ERA5 reanalysis data
# Must be have registered account at cds.climate.copernicus
# https://cds.climate.copernicus.eu/cdsapp#!/home
# File will be available through account once request is processed

def generate_ERA5_api_request(years, variables, months, days, times, latitude=(56, 50),
                              longitude=(-11, -5), format='grib'):

    """ API request for ERA5 reanalysis data

    years - list of years requested             (format ["2001", "2002", "2003",...])
    months - list of months                     (format ["01", "02", "03",...], str type)
    days - list of days                         (format ["01", "02", "03",...], str type)
    times - list of times (hourly)              (format ["00:00", "01:00", "02:00",...], str type)
    latitude - tuple of latitude range
    longitude - tuple of longitude range        (default is bounding box for Ireland)
    format - desired file format                only 'grib' or 'netcdf' available
    """

    import cdsapi

    c = cdsapi.Client()

    if format=='grib':
        format_extension = '.grib'
    else:
        format_extension = '.nc'

    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': variables,
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': [
                latitude[0], longitude[0], latitude[1],
                longitude[1],
            ],
            'format': format,
        },
        'download' + format_extension)



generate_ERA5_api_request()
