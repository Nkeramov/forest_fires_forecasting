# Yugra forest fires forecasting

## Description

This project presents an example of solving a data prediction problem using regression and correlation analysis.

Based on data on natural fires from 2000 to 2019 in the Khanty-Mansiysk Autonomous Okrug-Yugra, forecasted the number of fires, the total area of fires, and the total area of forest fires.

This forecast was not intended to predict the occurrence of fires in a particular area. The purpose of the forecast was to estimate the total area without reference to territories. This was required by the local authorities of the region to plan the amount of funding for extinguishing natural fires at the beginning of the calendar year, that is, long before the start of the fire season in the region (May-August).

Data on average monthly precipitation and temperature were also used for forecasting. Unfortunately, in open sources, only for some settlements (Khanty-Mansiysk, October, Leushi, Lariak and Ugut), it was possible to find a weather archive for the period from 2000 to 2020, broken down by months. Further, on the basis of statistics on fires, a reference settlement was determined, the weather values in which were taken as an average value for the region.

As practice has shown, the deviation of the forecast for the total area was no more than 15%.