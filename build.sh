#!/bin/bash

version=$1
echo "Version specified: $version"

# Exit in case of error
set -e

echo "Cleaning build directories..."
rm -rf build/
rm -rf dist/

echo "Building the wheel..."
python3 setup.py sdist bdist_wheel
