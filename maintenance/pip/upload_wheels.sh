#!/bin/bash

destination=$1

if [ -z "$destination" ] ; then
    echo "usage: upload_wheels.sh (testing|live)"
fi

if [ "$destination" == 'live' ] ; then
    twine upload dist/grond-* \
        --username="$PYPI_USERNAME" --password="$PYPI_PASSWORD" \
        --skip-existing --disable-progress-bar --comment='*grunz-grunz*'
else
    twine upload --repository-url https://test.pypi.org/legacy/ dist/grond-* \
        --username="$PYPI_USERNAME" --password="$PYPI_PASSWORD" \
        --skip-existing --disable-progress-bar --comment='*grunz-grunz*'
fi
