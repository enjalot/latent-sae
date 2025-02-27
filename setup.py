import os
import shutil
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Function to read the version from __version__.py
def get_version(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        for line in fp:
            if line.startswith('__version__'):
                # Executes the line of code and retrieves the __version__ variable
                ns = {}
                exec(line, ns)
                return ns['__version__']
    raise RuntimeError('Unable to find version string.')


version = get_version('latentsae/__version__.py')
print("building version", version)
setup(
    name='latentsae',
    version=version,
    description='LatentSAE: Training and inference for SAEs on embeddings',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/enjalot/latent-sae',
    project_urls={
        'Source': 'https://github.com/enjalot/latent-sae',
        'Tracker': 'https://github.com/enjalot/latent-sae/issues',
    },
    packages=find_packages(),
    install_requires=required,
    package_data={
        'latentsae.widgets.topk_vis': ['*.js', '*.css'],
        'latentsae.widgets.embedding_vis': ['*.js', '*.css'],
    },
    include_package_data=True,
    # entry_points={
    #     'console_scripts': [
    #     ],
    # },
    # include_package_data=True,
    # rest of your setup configuration...
)
