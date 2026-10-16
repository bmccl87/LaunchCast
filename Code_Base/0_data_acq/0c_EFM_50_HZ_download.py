import wget
import argparse

parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
parser.add_argument('--year', type=str, default='2019')
parser.add_argument('--url',type=str,default='https://https://kscweather.ksc.nasa.gov/')
args = parser.parse_args()
yr = args.year
url_arg = args.url

outdir = '/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/Field_Mill_50HZ_Feb2025/'
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

for mo in months:
    url=url_arg+'-'+yr+mo+'.zip'
    fname = yr+mo+'.zip'
    print(url)
    wget.download(url,out=outdir+fname)