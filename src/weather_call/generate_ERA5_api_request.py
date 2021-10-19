import cdsapi

def generate_ERA5_api_request(years, variables, months, days, times, latitude=(56, 50),
                              longitude=(-11, -5), format='grib'):

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
