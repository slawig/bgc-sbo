#!/usr/bin/env python
# -*- coding: utf8 -*

import os
from system.system import DATA_PATH, PYTHON_PATH, BACKUP_PATH, FIGURE_PATH, MEASUREMENT_DATA_PATH

PATH = os.path.join(DATA_PATH, 'SurrogateBasedOptimization')
PROGRAMM_PATH = os.path.join(PYTHON_PATH, 'SurrogateBasedOptimization')
PATH_BACKUP = os.path.join(BACKUP_PATH, 'SurrogateBasedOptimization', 'Optimization')
PATH_FIGURE = os.path.join(FIGURE_PATH, 'SurrogateBasedOptimization')
MEASUREMENT_DATA_WOA = os.path.join(MEASUREMENT_DATA_PATH, 'WOA')

DB_PATH = os.path.join(PATH, 'Database', 'SBO_Database.db')
DB_PATH_REORG = os.path.join(PATH, 'Database', 'SBO_Database_reorg.db')

PATH_OPTIMIZATION = 'SBO_{:0>4d}'
PATH_ITERATION = 'Iteration_{:0>4d}'
PATH_HIGH_FIDELITY_MODEL = 'HighFidelityModel'
PATH_LOW_FIDELITY_MODEL = 'LowFidelityModel_{:0>3d}'

PATTERN_LOGFILE = 'Logfile_SBO_{:0>3d}.txt'
PATTERN_JOBFILE = 'Jobfile_SBO_{:0>3d}.txt'
PATTERN_JOBOUTPUT = 'Joboutput_SBO_{:0>3d}.txt'

TARGET_TRACER_MODEL_YEARS = 10001
TARGET_TRACER_MODEL_TIMESTEP = 1
OPTIMIZATION_METHODS = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']

ANNID_MAX = 1

MISFIT_FUNCTIONS = ['TargetTracer', 'WOA_AnnualMean_PO4', 'WOA_MonthlyMean_PO4']
PATTERN_MEASUREMENT_ANNUAL_WOA_PO4 = 'woa13_all_pmean_01.petsc'
PATTERN_MEASUREMENT_MONTHLY_WOA_PO4 = 'woa13_all_p{:0>2d}_01.petsc'


PATTERN_FIGURE_COSTFUNCTION = 'Costfunction.{:s}.OptimizationId_{:0>4d}.pdf'
PATTERN_FIGURE_STEPSIZENORM = 'StepSizeNorm.{:s}.OptimizationId_{:0>4d}.pdf'
PATTERN_FIGURE_PARAMETERCONVERGENCE = 'ParameterConvergence.{:s}.Parameter_{:d}.OptimizationId_{:0>4d}.pdf'
PATTERN_FIGURE_ANNUALCYCLE = 'AnnualCycle.{:s}.Latitude_{}.Longitude_{}_Depth_{:d}.OptimizationId_{:0>4d}.pdf'
PATTERN_FIGURE_ANNUALCYCLEPARAMETER = 'AnnualCycleParameter.{:s}.Latitude_{}.Longitude_{}_Depth_{:d}.OptimizationId_{:0>4d}.pdf'
PATTERN_FIGURE_SURFACE_TARGET_TRACER = 'Surface_OptimizationId_{:0>4d}.Projection_{:s}.{:s}.Depth_{:d}.Tracer_{:s}.TargetTracer.pdf'
PATTERN_FIGURE_SURFACE = 'Surface.OptimizationId_{:0>4d}.Projection_{:s}.{:s}.Depth_{:d}.Tracer_{:s}.{:s}.Difference_{}.RelError_{}.pdf'
PATTERN_FIGURE_SURFACE_PARAMETER = 'Surface.OptimizationId_{:0>4d}.Projection_{:s}.{:s}.Depth_{:d}.Tracer_{:s}.ParameterId_{:d}.{:s}.Difference_{}.RelError_{}.pdf'
PATTERN_FIGURE_SURFACE_LOWFIDELITYMODEL = 'Surface.OptimizationId_{:0>4d}.Projection_{:s}.{:s}.Depth_{:d}.Tracer_{:s}.LowFidelityModelDifference.ParameterIds_{:d}_{:d}.RelError_{}.pdf'

PATTERN_BACKUP_FILENAME = 'SBO_Backup_OptimizationId_{:0>4d}.tar.{}'
PATTERN_BACKUP_LOGFILE = 'SBO_Backup_OptimizationId_{:0>4d}_Backup_{}_Remove_{}_Restore_{}.log'
COMPRESSION = 'bz2'
COMPRESSLEVEL = 9

