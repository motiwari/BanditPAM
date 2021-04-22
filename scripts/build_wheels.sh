for PYBIN in /opt/python/*/bin; do
  "${PYBIN}/python" setup.py sdist bdist_wheel
done
mkdir /io/wheelhouse
cp dist/*.gz /io/wheelhouse
for whl in dist/*.whl; do
  auditwheel repair "$whl" -w /io/wheelhouse
done