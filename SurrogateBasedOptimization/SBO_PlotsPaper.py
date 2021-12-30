#!/usr/bin/env python
# -*- coding: utf8 -*

import os

import metos3dutil.metos3d.constants as Metos3d_Constants
from plot.SurrogateBasedOptimizationPlot import SurrogateBasedOptimizationPlot
import sbo.constants as SBO_Constants

#Global variables
PATH_FIGURE = os.path.join('/gxfs_home', 'cau', 'sunip350', 'Daten', 'Figures', 'SurrogateBasedOptimization', 'PaperData')

def main(optimizationIdList=[(39, 'N-DOP'), (111, 'N'), (207, 'N'), (233, 'N')], nodes=1, fontsize=9):
    """
    Create the plots for the surrogate based optimization paper

    Parameter
    ---------
    optimizationIdList : list [tuple [int, str]], default: [(175, 'N-DOP')]
        Plot the results for all ids of the given optimizations runs. The tuple
        contains the optimizationId (int) and the model (str).
    nodes : int, default: 1
        Number of nodes on the high performance cluster
    fontsize : int, default: 10
        Fontsize in the plots
    """
    assert type(optimizationIdList) is list
    assert type(nodes) is int and nodes > 0
    assert type(fontsize) is int and 0 < fontsize

    os.makedirs(PATH_FIGURE, exist_ok=True)
    
    kwargs = {}
    kwargs[39] = {}
    kwargs[39]['Costfunction'] = {'subplot_adjust': {'left': 0.16, 'bottom': 0.195, 'right': 0.985, 'top': 0.995}, 'xticks': [0, 2, 4, 6, 8, 10]}
    kwargs[39]['StepSizeNorm'] = {'subplot_adjust': {'left': 0.245, 'bottom': 0.195, 'right': 0.985, 'top': 0.995}, 'xticks': [0, 2, 4, 6, 8, 10]}
    kwargs[39]['ParameterConvergence'] = {}
    kwargs[39]['ParameterConvergence']['0'] = {'subplot_adjust': {'left': 0.275, 'bottom': 0.2225, 'right': 0.975, 'top': 0.9825}, 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0.02, 0.03, 0.04, 0.05], 'legend': True}
    kwargs[39]['ParameterConvergence']['1'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.975, 'top': 0.9825}, 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [2.0, 2.5, 3.0, 3.5], 'legend': False}
    kwargs[39]['ParameterConvergence']['2'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.975, 'top': 0.81}, 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'legend': False}
    kwargs[39]['ParameterConvergence']['3'] = {'subplot_adjust': {'left': 0.2225, 'bottom': 0.2225, 'right': 0.975, 'top': 0.985}, 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [25, 26, 27, 28, 29, 30], 'legend': False}
    kwargs[39]['ParameterConvergence']['4'] = {'subplot_adjust': {'left': 0.2325, 'bottom': 0.2225, 'right': 0.975, 'top': 0.995}, 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65], 'legend': False}
    kwargs[39]['ParameterConvergence']['5'] = {'subplot_adjust': {'left': 0.275, 'bottom': 0.2225, 'right': 0.975, 'top': 0.995}, 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50], 'legend': False}
    kwargs[39]['ParameterConvergence']['6'] = {'subplot_adjust': {'left': 0.25, 'bottom': 0.2225, 'right': 0.975, 'top': 0.9725}, 'xticks': [0, 2, 4, 6, 8, 10], 'yticks': [0.800, 0.850, 0.900], 'legend': False}
    kwargs[39]['AnnualCycle'] = {'subplot_adjust': {'left': 0.20, 'bottom': 0.245, 'right': 0.995, 'top': 0.995}, 'iterations': [0, 2, 5, 8, 10], 'layer': 0}
    kwargs[39]['Surface'] = {'plotTargetTracer': True}

    kwargs[111] = {}
    kwargs[111]['Costfunction'] = {'subplot_adjust': {'left': 0.16, 'bottom': 0.195, 'right': 0.995, 'top': 0.995}, 'xticks': [0, 2, 4, 6, 8, 10, 12]}
    kwargs[111]['StepSizeNorm'] = {'subplot_adjust': {'left': 0.245, 'bottom': 0.195, 'right': 0.995, 'top': 0.995}, 'xticks': [2, 4, 6, 8, 10, 12]}
    kwargs[111]['ParameterConvergence'] = {}
    kwargs[111]['ParameterConvergence']['0'] = {'subplot_adjust': {'left': 0.31, 'bottom': 0.2225, 'right': 0.995, 'top': 0.9825}, 'xticks': [0, 2, 4, 6, 8, 10, 12], 'yticks': [0.02, 0.025, 0.03, 0.035, 0.04], 'legend': True}
    kwargs[111]['ParameterConvergence']['1'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.995, 'top': 0.9825}, 'xticks': [0, 2, 4, 6, 8, 10, 12], 'yticks': [2.0, 2.5, 3.0, 3.5], 'legend': False}
    kwargs[111]['ParameterConvergence']['2'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.995, 'top': 0.81}, 'xticks': [0, 2, 4, 6, 8, 10, 12], 'yticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'legend': False}
    kwargs[111]['ParameterConvergence']['3'] = {'subplot_adjust': {'left': 0.2225, 'bottom': 0.2225, 'right': 0.995, 'top': 0.985}, 'xticks': [0, 2, 4, 6, 8, 10, 12], 'yticks': [25, 26, 27, 28, 29, 30], 'legend': False}
    kwargs[111]['ParameterConvergence']['4'] = {'subplot_adjust': {'left': 0.2525, 'bottom': 0.2225, 'right': 0.995, 'top': 0.995}, 'xticks': [0, 2, 4, 6, 8, 10, 12], 'yticks': [0.78, 0.80, 0.82, 0.84, 0.86], 'legend': False}
    kwargs[111]['AnnualCycle'] = {'subplot_adjust': {'left': 0.225, 'bottom': 0.245, 'right': 0.995, 'top': 0.995}, 'iterations': [0, 4, 7, 10, 13], 'layer': 0}
    kwargs[111]['Surface'] = {'plotTargetTracer': True}
    kwargs[111]['AnnualCycleParameter'] = {'subplot_adjust': {'left': 0.2, 'bottom': 0.245, 'right': 0.995, 'top': 0.995}}
    kwargs[111]['SurfaceParameter'] = {}

    kwargs[207] = {}
    kwargs[207]['Costfunction'] = {'subplot_adjust': {'left': 0.16, 'bottom': 0.21, 'right': 0.985, 'top': 0.995}, 'acceptedIterationOnly': False, 'xlabel': 'High-fidelity model evaluations', 'ylabel': '$J(\mathbf{y}_f(\mathbf{u}))$'}
    kwargs[207]['StepSizeNorm'] = {'subplot_adjust': {'left': 0.245, 'bottom': 0.195, 'right': 0.995, 'top': 0.995}}
    kwargs[207]['ParameterConvergence'] = {}
    kwargs[207]['ParameterConvergence']['0'] = {'subplot_adjust': {'left': 0.31, 'bottom': 0.2225, 'right': 0.995, 'top': 0.9825}, 'legend': True}
    kwargs[207]['ParameterConvergence']['1'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.995, 'top': 0.9825}, 'legend': False}
    kwargs[207]['ParameterConvergence']['2'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.995, 'top': 0.81}, 'legend': False}
    kwargs[207]['ParameterConvergence']['3'] = {'subplot_adjust': {'left': 0.2225, 'bottom': 0.2225, 'right': 0.995, 'top': 0.985}, 'legend': False}
    kwargs[207]['ParameterConvergence']['4'] = {'subplot_adjust': {'left': 0.2525, 'bottom': 0.2225, 'right': 0.995, 'top': 0.995}, 'legend': False}
    kwargs[207]['AnnualCycle'] = {'subplot_adjust': {'left': 0.225, 'bottom': 0.245, 'right': 0.995, 'top': 0.995}, 'iterations': [0, 1], 'layer': 0}
    kwargs[207]['Surface'] = {'plotTargetTracer': True}
    kwargs[207]['AnnualCycleParameter'] = {'subplot_adjust': {'left': 0.2, 'bottom': 0.245, 'right': 0.995, 'top': 0.995}}
    kwargs[207]['SurfaceParameter'] = {}

    kwargs[233] = {}
    kwargs[233]['Costfunction'] = {'subplot_adjust': {'left': 0.16, 'bottom': 0.195, 'right': 0.985, 'top': 0.995}, 'xticks': [0, 5, 10, 15, 20]}
    kwargs[233]['StepSizeNorm'] = {'subplot_adjust': {'left': 0.245, 'bottom': 0.195, 'right': 0.985, 'top': 0.995}, 'xticks': [5, 10, 15, 20]}
    kwargs[233]['ParameterConvergence'] = {}
    kwargs[233]['ParameterConvergence']['0'] = {'subplot_adjust': {'left': 0.31, 'bottom': 0.2225, 'right': 0.975, 'top': 0.9825}, 'xticks': [0, 5, 10, 15, 20], 'yticks': [0.02, 0.025, 0.03, 0.035, 0.04], 'legend': True}
    kwargs[233]['ParameterConvergence']['1'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.975, 'top': 0.9825}, 'xticks': [0, 5, 10, 15, 20], 'yticks': [2.0, 2.5, 3.0, 3.5], 'legend': False}
    kwargs[233]['ParameterConvergence']['2'] = {'subplot_adjust': {'left': 0.2425, 'bottom': 0.2225, 'right': 0.975, 'top': 0.81}, 'xticks': [0, 5, 10, 15, 20], 'yticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'legend': False}
    kwargs[233]['ParameterConvergence']['3'] = {'subplot_adjust': {'left': 0.2225, 'bottom': 0.2225, 'right': 0.975, 'top': 0.985}, 'xticks': [0, 5, 10, 15, 20], 'yticks': [25, 26, 27, 28, 29, 30], 'legend': False}
    kwargs[233]['ParameterConvergence']['4'] = {'subplot_adjust': {'left': 0.2525, 'bottom': 0.2225, 'right': 0.975, 'top': 0.995}, 'xticks': [0, 5, 10, 15, 20], 'yticks': [0.78, 0.80, 0.82, 0.84, 0.86], 'legend': False}
    kwargs[233]['AnnualCycle'] = {'subplot_adjust': {'left': 0.225, 'bottom': 0.245, 'right': 0.995, 'top': 0.995}, 'iterations': [0, 5, 10, 15, 20], 'layer': 0}
    kwargs[233]['Surface'] = {'plotTargetTracer': True}
    kwargs[233]['AnnualCycleParameter'] = {'subplot_adjust': {'left': 0.2, 'bottom': 0.245, 'right': 0.995, 'top': 0.995}, 'positionList': [(0.0, 90.0), (30.9375, -30.9375), (30.9375, -120.9375)]}
    kwargs[233]['SurfaceParameter'] = {}

    for optimizationId, metos3dModel in optimizationIdList:
        sboPlot = SurrogateBasedOptimizationPlot(optimizationId, nodes=nodes, orientation='lc2', fontsize=fontsize)

        #Plot the cost function values
        plotCostfunction(sboPlot, optimizationId, metos3dModel, fontsize=fontsize, **kwargs[optimizationId]['Costfunction'])

        #Plot the step size norm
        plotStepSizeNorm(sboPlot, optimizationId, metos3dModel, fontsize=fontsize, **kwargs[optimizationId]['StepSizeNorm'])

        #Plot the parameter convergence
        plotParameterConvergence(sboPlot, optimizationId, metos3dModel, fontsize=fontsize, **kwargs[optimizationId]['ParameterConvergence'])

        #Plot the annual cycle of the tracer concentration
        plotAnnualCycle(sboPlot, optimizationId, metos3dModel, fontsize=fontsize, **kwargs[optimizationId]['AnnualCycle'])

        #Plot the surface concentration
        plotSurface(sboPlot, metos3dModel, fontsize=fontsize, **kwargs[optimizationId]['Surface'])
        
        #Plot the surface concentration of the surrogate construction
        if 'SurfaceParameter' in kwargs[optimizationId]:
            plotSurfaceParameter(sboPlot, metos3dModel, fontsize=fontsize, **kwargs[optimizationId]['SurfaceParameter'])
        
        #Crate table of the parameter vectors
        tableStr = tableParametervector(sboPlot, optimizationId, metos3dModel, **kwargs[optimizationId]['parameterTable'])
        print('Table parameter vectors for {} with Id {}\n{}'.format(metos3dModel, optimizationId, tableStr))

        #Crate the annual cylce of the tracer concentration including the low-fidelty model and surrogate
        if 'AnnualCycleParameter' in kwargs[optimizationId]:
            plotAnnualCycleParameter(sboPlot, optimizationId, metos3dModel, fontsize=fontsize, **kwargs[optimizationId]['AnnualCycleParameter'])



def plotCostfunction(sboPlot, optimizationId, metos3dModel, orientation='lc2', fontsize=10, **kwargs):
    """
    Plot the cost function values

    Parameter
    ---------
    sboPlot
    optimizationId : int
        OptimizationId of the optimization run
    metos3dModel : str
        Biogeochemical model
    orientation : str, default: 'lc2'
        Str to define the size of the plots
    fontsize : int, default: 10
        Fontsize in the plots
    **kwargs : dict
        Additional keyword arguments with keys:
        subplot_adjust : dict [str, float], optional
            Adjustment of the subplot using the keys left, bottom, right
            and top

    Notes
    -----
    Save the figure
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize

    #Parse keyword arguments
    subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.16, 'bottom': 0.195, 'right': 0.985, 'top': 0.995}
    acceptedOnly = kwargs['acceptedIterationOnly'] if 'acceptedIterationOnly' in kwargs and type(kwargs['acceptedIterationOnly']) is bool else True

    sboPlot._init_plot(orientation=orientation, fontsize=fontsize)
    sboPlot.plot_costfunction(acceptedOnly=acceptedOnly, **kwargs)
    sboPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])
    filename = os.path.join(PATH_FIGURE, SBO_Constants.PATTERN_FIGURE_COSTFUNCTION.format(metos3dModel, optimizationId))
    sboPlot.savefig(filename)
    sboPlot.close_fig()


def plotStepSizeNorm(sboPlot, optimizationId, metos3dModel, orientation='lc2', fontsize=10, **kwargs):
    """
    Plot the step size norm

    Parameter
    ---------
    sboPlot
    optimizationId : int
        OptimizationId of the optimization run
    metos3dModel : str
        Biogeochemical model
    orientation : str, default: 'lc2'
        Str to define the size of the plots
    fontsize : int, default: 10
        Fontsize in the plots
    **kwargs : dict
        Additional keyword arguments with keys:
        subplot_adjust : dict [str, float], optional
            Adjustment of the subplot using the keys left, bottom, right
            and top

    Notes
    -----
    Save the figure
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize

    #Parse keyword arguments
    subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.16, 'bottom': 0.195, 'right': 0.985, 'top': 0.995}

    sboPlot._init_plot(orientation=orientation, fontsize=fontsize)
    sboPlot.plot_stepsize_norm(**kwargs)
    sboPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])
    filename = os.path.join(PATH_FIGURE, SBO_Constants.PATTERN_FIGURE_STEPSIZENORM.format(metos3dModel, optimizationId))
    sboPlot.savefig(filename)
    sboPlot.close_fig()


def plotParameterConvergence(sboPlot, optimizationId, metos3dModel, orientation='lc3', fontsize=10, **kwargs):
    """
    Plot the parameter convergence for each model parameter

    Parameter
    ---------
    sboPlot
    optimizationId : int
        OptimizationId of the optimization run
    metos3dModel : str
        Biogeochemical model
    orientation : str, default: 'lc2'
        Str to define the size of the plots
    fontsize : int, default: 10
        Fontsize in the plots
    **kwargs : dict
        Additional keyword arguments with keys:
        subplot_adjust : dict [str, float], optional
            Adjustment of the subplot using the keys left, bottom, right
            and top

    Notes
    -----
    Save the figure
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize

    for i in range(Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[metos3dModel]):
        #Parse keyword arguments
        subplot_adjust = kwargs[str(i)]['subplot_adjust'] if 'subplot_adjust' in kwargs[str(i)] and type(kwargs[str(i)]['subplot_adjust']) is dict and 'left' in kwargs[str(i)]['subplot_adjust'] and 'bottom' in kwargs[str(i)]['subplot_adjust'] and 'right' in kwargs[str(i)]['subplot_adjust'] and 'top' in kwargs[str(i)]['subplot_adjust'] else {'left': 0.16, 'bottom': 0.195, 'right': 0.985, 'top': 0.995}

        sboPlot._init_plot(orientation=orientation, fontsize=fontsize)
        sboPlot.plot_parameter_convergence(parameterIndex=i, **kwargs[str(i)])
        sboPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])
        filename = os.path.join(PATH_FIGURE, SBO_Constants.PATTERN_FIGURE_PARAMETERCONVERGENCE.format(metos3dModel, i+1, optimizationId))
        sboPlot.savefig(filename)
        sboPlot.close_fig()


def plotAnnualCycle(sboPlot, optimizationId, metos3dModel, orientation='lc2', fontsize=10, **kwargs):
    """
    Plot the annual cycle of the tracer concentrations for different iterates

    Parameter
    ---------
    sboPlot
    optimizationId : int
        OptimizationId of the optimization run
    metos3dModel : str
        Biogeochemical model
    orientation : str, default: 'lc2'
        Str to define the size of the plots
    fontsize : int, default: 10
        Fontsize in the plots
    **kwargs : dict
        Additional keyword arguments with keys:
        subplot_adjust : dict [str, float], optional
            Adjustment of the subplot using the keys left, bottom, right
            and top

    Notes
    -----
    Save the figure
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize

    #Parse keyword arguments
    subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.155, 'bottom': 0.16, 'right': 0.995, 'top': 0.995}
    positionList = kwargs['postionList'] if 'positionList' in kwargs and type(kwargs['positonList']) is list else [(30.9375, -30.9375), (30.9375, -120.9375), (0.0, 90.0)]
    depth = kwargs['layer'] if 'layer' in kwargs and type(kwargs['layer']) is int else 0
    iterationList = kwargs['iterations'] if 'iterations' in kwargs and type(kwargs['iterations']) is list else [0, 2, 5, 8, 10]
    runMetos3d = kwargs['runMetos3dFlag'] if 'runMetos3dFlag' in kwargs and type(kwargs['runMetos3dFlag']) is bool else False
    plotFigure = kwargs['plotFigureFlag'] if 'plotFigureFlag' in kwargs and type(kwargs['plotFigureFlag']) is bool else True    

    i = 1
    for latitude, longitude in positionList:
        if i == len(positionList):
            kwargs['legend'] = True
            subplot_adjust = {'left': 0.15, 'bottom': 0.245, 'right': 0.765, 'top': 0.995}
            orientation = 'lc2long'
        else:
            kwargs['legend'] = False 
            orientation = 'lc2short'
        sboPlot._init_plot(orientation=orientation, fontsize=fontsize)
        sboPlot.plot_annual_cycles(iterationList, latitude=latitude, longitude=longitude, depth=depth, runMetos3d=runMetos3d, plot=plotFigure, remove=False, **kwargs)
        sboPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])
        filename = os.path.join(PATH_FIGURE, SBO_Constants.PATTERN_FIGURE_ANNUALCYCLE.format(metos3dModel, latitude, longitude, depth, optimizationId))
        if plotFigure:
            sboPlot.savefig(filename)
        sboPlot.close_fig()
        i += 1


def plotSurface(sboPlot, metos3dModel, orientation='lc3', fontsize=10, **kwargs):
    """
    Plot the annual cycle of the tracer concentrations for different iterates

    Parameter
    ---------
    sboPlot
    metos3dModel : str
        Biogeochemical model
    orientation : str, default: 'lc2'
        Str to define the size of the plots
    fontsize : int, default: 10
        Fontsize in the plots
    **kwargs : dict
        Additional keyword arguments with keys:
        subplot_adjust : dict [str, float], optional
            Adjustment of the subplot using the keys left, bottom, right
            and top

    Notes
    -----
    Save the figure
    """
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize

    #Parse keyword arguments
    depth = kwargs['depth'] if 'depth' in kwargs and type(kwargs['depth']) is int else 0
    tracer = kwargs['tracer'] if 'tracer' in kwargs and type(kwargs['tracer']) is str else Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel][0]
    tracerDifference = kwargs['tracerDifference'] if 'tracerDifference' in kwargs and type(kwargs['tracerDifference']) is bool else False
    relativeError = kwargs['relativeError'] if 'relativeError' in kwargs and type(kwargs['relativeError']) is bool else False
    plotTargetTracer = kwargs['plotTargetTracer'] if 'plotTargetTracer' in kwargs and type(kwargs['plotTargetTracer']) is bool else False

    sboPlot.plot_surface(depth=depth, orientation=orientation, fontsize=fontsize, tracer=tracer, tracerDifference=tracerDifference, relativeError=relativeError, plotTargetTracer=plotTargetTracer, vmin=0.0, vmax=2.0)


def tableParametervector(sboPlot, optimizationId, metos3dModel, **kwargs):
    """
    Create a string for a table in LaTeX with the parameter vectors of the iterations in the SBO run

    Parameter
    ---------
    optimizationId : int
        OptimizationId of the optimization run
    metos3dModel : str
        Biogeochemical model
    **kwargs : dict
        Additional keyword argument with keys:
        iterationList : list [int], optional
            List with numbers of iteration to include the parameter vectors into the table
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

    #Parse keyword arguments
    iterationList = kwargs['iterationList'] if 'iterationList' in kwargs and type(kwargs['iterationList']) is list else []

    #Read parameter vector for each iteration
    parameterIds = sboPlot._sboDB.get_iteration_parameterIds(optimizationId)

    #Create tableStr
    tableStr = '\\hline\niterate & ${:s}$'.format('$ & $'.join(map(str, Metos3d_Constants.PARAMETER_NAMES_LATEX[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]])))
    tableStr += ' \\\\\n\\hline\n'

    for (iteration, parameterId) in parameterIds:
        if len(iterationList) == 0 or iteration in iterationList:
            tableStr += '$\\mathbf{:s}_{:s}$ &'.format('{u}', '{' + '{:d}'.format(iteration) + '}')
            parameterList = [' {:.3f} '.format(p) for p in sboPlot._sboDB.get_parameter(parameterId, metos3dModel)]
            tableStr += '{:s} \\\\\n'.format(' & '.join(map(str, parameterList)))

    #Target tracer
    tableStr += '$\\mathbf{:s}_d$ & {:s} \\\\\n'.format('{u}', ' & '.join(map(str, [' {:.3f} '.format(p) for p in sboPlot._sboDB.get_target_parameter(optimizationId)])))
    tableStr += '\\hline\n'

    return tableStr


def plotAnnualCycleParameter(sboPlot, optimizationId, metos3dModel, fontsize=10, **kwargs):
    """
    Plot the annual cycle of the tracer concentrations for the given parameter
    including the low-fidelity model and the surrogate

    Parameter
    ---------
    sboPlot
    optimizationId : int
        OptimizationId of the optimization run
    metos3dModel : str
        Biogeochemical model
    fontsize : int, default: 10
        Fontsize in the plots
    **kwargs : dict
        Additional keyword arguments with keys:
        subplot_adjust : dict [str, float], optional
            Adjustment of the subplot using the keys left, bottom, right
            and top
        parameterIdList : list [int], optional
            List of parameterIds
        runMetos3dFlag : bool
            If True, calculate the low-fidelity, high-fidelity model and the
            surrogate
        plotFigureFlag : bool
            If True, create the figure

    Notes
    -----
    Save the figure
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(fontsize) is int and 0 < fontsize

    #Parse keyword arguments
    subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.155, 'bottom': 0.16, 'right': 0.995, 'top': 0.995}
    positionList = kwargs['positionList'] if 'positionList' in kwargs and type(kwargs['positionList']) is list else [(30.9375, -30.9375), (30.9375, -120.9375), (0.0, 90.0)]
    depth = kwargs['layer'] if 'layer' in kwargs and type(kwargs['layer']) is int else 0
    parameterIdList = kwargs['parameterIdList'] if 'parameterIdList' in kwargs and type(kwargs['parameterIdList']) is list else [0, 1152, 1153]
    labelList = kwargs['labelList'] if 'labelList' in kwargs and type(kwargs['labelList']) is list else [r'\mathbf{{u}}_{{0}}', r'\bar{{\mathbf{{u}}}}', r'\tilde{{\mathbf{{u}}}}']
    runMetos3d = kwargs['runMetos3dFlag'] if 'runMetos3dFlag' in kwargs and type(kwargs['runMetos3dFlag']) is bool else False
    plotFigure = kwargs['plotFigureFlag'] if 'plotFigureFlag' in kwargs and type(kwargs['plotFigureFlag']) is bool else True    

    i = 1
    for latitude, longitude in positionList:
        if i == len(positionList):
            kwargs['legend'] = True
            subplot_adjust = {'left': 0.15, 'bottom': 0.245, 'right': 0.78, 'top': 0.995}
            orientation = 'lc2long'
        else:
            kwargs['legend'] = False 
            orientation = 'lc2short'
        sboPlot._init_plot(orientation=orientation, fontsize=fontsize)
        sboPlot.plot_annual_cycles_parameter(parameterIdList=parameterIdList, labelList=labelList, plotSurrogate=True, latitude=latitude, longitude=longitude, depth=depth, runMetos3d=runMetos3d, plot=plotFigure, remove=False, **kwargs)
        sboPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])
        filename = os.path.join(PATH_FIGURE, SBO_Constants.PATTERN_FIGURE_ANNUALCYCLEPARAMETER.format(metos3dModel, latitude, longitude, depth, optimizationId))
        if plotFigure:
            sboPlot.savefig(filename)
        sboPlot.close_fig()
        i += 1


def plotSurfaceParameter(sboPlot, metos3dModel, orientation='lc3', fontsize=10, **kwargs):
    """
    Plot the annual cycle of the tracer concentrations for different iterates

    Parameter
    ---------
    sboPlot
    metos3dModel : str
        Biogeochemical model
    orientation : str, default: 'lc2'
        Str to define the size of the plots
    fontsize : int, default: 10
        Fontsize in the plots
    **kwargs : dict
        Additional keyword arguments with keys:
        subplot_adjust : dict [str, float], optional
            Adjustment of the subplot using the keys left, bottom, right
            and top

    Notes
    -----
    Save the figure
    """
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize

    #Parse keyword arguments
    depth = kwargs['depth'] if 'depth' in kwargs and type(kwargs['depth']) is int else 0
    tracer = kwargs['tracer'] if 'tracer' in kwargs and type(kwargs['tracer']) is str else Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel][0]
    tracerDifference = kwargs['tracerDifference'] if 'tracerDifference' in kwargs and type(kwargs['tracerDifference']) is bool else True
    relativeError = kwargs['relativeError'] if 'relativeError' in kwargs and type(kwargs['relativeError']) is bool else False
    parameterIdList = kwargs['parameterIdList'] if 'parameterIdList' in kwargs and type(kwargs['parameterIdList']) is list else [0, 1152, 1153]
    extend = kwargs['extend'] if 'extend' in kwargs and type(kwargs['extend']) is str else 'both'
    vmin = kwargs['vmin'] if 'vmin' in kwargs and type(kwargs['vmin']) is float else -1.0
    vmax = kwargs['vmax'] if 'vmin' in kwargs and type(kwargs['vmax']) is float else 1.0

    sboPlot.plot_surface_parameter(parameterIdList=parameterIdList, plotHighFidelityModel=not tracerDifference, tracerDifference=tracerDifference, relativeError=relativeError, depth=depth, orientation=orientation, fontsize=fontsize, tracer=tracer, plot=True, vmin=vmin, vmax=vmax, extend=extend)


if __name__ == '__main__':
    main()

