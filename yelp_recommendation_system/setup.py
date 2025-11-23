from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent

# read reqs from requirements.txt
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='yelp-hybrid-recommender',
    version='1.0.0',
    author='Jacob Strickland',
    description='A hybrid recommendation system for Yelp restaurant recommendations',
    packages=find_packages(where='.'),
    python_requires='>=3.7',
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)

