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
    package_dir={'sbayes': 'sbayes'},
    packages=find_packages(),
    platforms='any',
    include_package_data=True,
    package_data={
        'sbayes.config': ['default_config_plot.json'],
        'sbayes.maps': ['land.geojson', 'rivers_lake.geojson']
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
    ],
    entry_points={
        'console_scripts': [
            'sblot = sbayes.plot:cli',
        ]
    }
)

