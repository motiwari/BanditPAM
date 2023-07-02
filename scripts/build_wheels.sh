#!/bin/bash

for PYBIN in /opt/python/*3*/bin; do
  "${PYBIN}/python" setup.py sdist bdist_wheel
done
mkdir -p /io/wheelhouse
cp dist/*.gz /io/wheelhouse
for whl in dist/*.whl; do
  auditwheel repair "$whl" -w /io/wheelhouse
done
