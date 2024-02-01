from setuptools import setup, find_packages

setup(
    name="sBlot",
    version="1.0",
    description="Python library for visualization of sBayes results.",
    author="Anjing Zhang, Nico Neureiter, Peter Ranacher",
    author_email="nico.neureiter@gmail.com",
    keywords='data linguistics',
    license='GPLv3',
    url="https://github.com/maeva-jing/sBlot",
    package_dir={'sblot': 'sblot'},
    packages=find_packages(),
    platforms='any',
    include_package_data=True,
    package_data={
        'sblot.config': ['default_config_plot.json'],
        'sblot.maps': ['land.geojson', 'rivers_lake.geojson']
    },
    install_requires=[
        "descartes",
        "geopandas",
        "matplotlib",
        "numpy",
        "pygeos",
        "pandas",
        "pyproj",
        "scipy",
        "Shapely",
        "seaborn",
        "setuptools",
        "cartopy",
        "typing_extensions",
        "pydantic",
        "ruamel.yaml",
        "rasterio",
        "webcolors"
    ],
    entry_points={
        'console_scripts': [
            'sblot = sblot.plot:cli',
        ]
    }
)

