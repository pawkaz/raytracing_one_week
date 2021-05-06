#!/bin/bash

clang++ main.cpp -O3 -o a.out && ./a.out > test.ppm && convert test.ppm test.jpg