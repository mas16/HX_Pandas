# HX_Pandas: Fitting 1H-15N Amide Hydrogen Exchange Rates in Proteins from NMR Data
by MAS

## Introduction
NMR is a powerful technique for obtaining quantitative information about the structural dynamics of proteins. This script will fit hydrogen exchange data collected from a series of 2D 1H-15N heteronuclear single quantum coherence spectra.

This script will read raw, uncleaned nmrPipe output ```.tab``` (flat file) and extract relevant data for fitting serially collected HX data. Number of data points used in the fitting can be arbitrary.

Data are fit to standard three parameter exponential decay of form:

$I(t) = I(0) * exp(-Rt) + b$

where I is signal intensity, R is the decay rate, and b is the offset.

Fitting is done via non linear least squares optimization using scipy.

Generators are used to minimize the amount of data stored in memory.

## Libraries
Please install the following libraries/packages before running
```numpy```    
```matplotlib```   
```scipy```   
```seaborn```     

## Input 
Input data should be the raw ```nmrPipe``` ```.tab``` file
