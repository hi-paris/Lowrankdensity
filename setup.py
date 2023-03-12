"""The setup script."""

from setuptools import setup, find_packages

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = []
test_requirements = [ ]

setup(
    author="Julien Chhor, Olga Klopp, Alexandre Tsybakov",
    author_email='gaetan.brison@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    install_requires=["numpy","scipy","sklearn"],
    license="MIT license",
    include_package_data=True,
    keywords='DensLowRank',
    name='DensLowRank',
    packages=find_packages(include=['DensLowRank', 'DensLowRank.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/hi-paris/DensLowRank',
    version='0.0.1',
    zip_safe=False,
)
