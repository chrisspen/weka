#!/bin/bash
FILES=`git status --porcelain | grep -E "*\.py$" | grep -v migration | grep -v " D " | awk '{print $2}'`
echo "Checking: $FILES"
pylint --rcfile=pylint.rc $FILES
