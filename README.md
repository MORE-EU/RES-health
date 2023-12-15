RES-health
==========
RES health monitoring toolkit
(initial integrated version of the platform) <br>


RES-health is a Python library for Pattern Extraction and Complex Event Detection in time series from Renewable Energy Sources (RES). RES-health implements various analytical tools with an emphasis on real-world use cases with RES data. In short, the library contains tools for deviation/behavior detection, detection of failures, forecasting, pattern extraction. Further details on the implemented tools and associated modules follow.
  
Description
===========	
For the development process, several python libraries will be used. In particular: [STUMPY](https://stumpy.readthedocs.io/en/latest/), [MatrixProfile](https://matrixprofile.docs.matrixprofile.org/), [Pyscamp](https://pypi.org/project/pyscamp/) and [Scikit-Learn](https://scikit-learn.org/stable/) . The aforementioned libraries are implemented in [python3](https://www.python.org/download/releases/3.0/) (or provide python3 bindings to C++ code), thus the Event Detections module will adopt the same programming language. An installation of the [CUDA toolkit8](https://developer.nvidia.com/cuda-toolkit) is necessary for deploying the GPU-accelerated versions of the aforementioned libraries. 
The module consists of the following steps:

1. **Input/Output**

  This group of utilities includes all functions which are responsible for loading and storing time series to the disk. Moreover, it includes functions for loading/storing necessary metadata which are produced by any of the methods of this module and can be further processed.

2. **Preprocessing**

  Utilities regarding the handling and processing of the input data. Such functions handle filtering, resampling, adding noise, normalization and filling out missing values.
  
3. **Statistics**

  Utilities regarding the computation of useful statistics about the dataset, that implement variations of standard error measures, evaluating the performance of a regression   model, ranking time series.
   
4. **Plotting**   

  Functions that are used for the plotting of the results are presented, also include functions for plotting auxiliary information and helper functions for switching between     different data formats. 
  
5. **Basic pattern extraction**

  In this section, we discuss the implementation of functions related to pattern extraction. This includes functions for computing a matrix profile, but also functions related to the post-processing of a matrix profile. The latter refers to functions for discovering motifs/discords or segmenting a time series.
   
6. **Similarity**
  
  Utilities related to similarity search refer to functions which are responsible for performing proximity queries to a dataset. Queries that can be handled include finding the k-nearest neighbors of a given vector or time series. One example of such a query is computing the top k-nearest neighbors of a given subsequence in a multivariate time series, among all (or a subset of) subsequences in the same time series.

7. **Learning**
  
  In this set of utilities, we implement functions that build upon standard machine learning algorithms. In particular, functions contained in this module, handle tasks related 
  to approximating a time series by means of a regression model. Those functions are needed, for example, when we are trying to discover trends or changepoints. 

RES-health tools
================
  The execution of the project starts with the presentation of the tools. 

1. **Deviation Detection**

  In this section, we present a tool for detecting segments of a time series where the behavior of a given target variable deviates much from the usual. We assume that certain 
  segments of the time series are labelled, representing the usual behavior of the variable in question. In this tool the user can find the list of variants that took place to implement it.

  * Self-supervised changepoint detection: This refers to our method for detecting changepoints in one variable’s behavior as follows: given a set of segments of the input time series, our method decides for each one of those segments whether it contains one changepoint
  
  * Semi-supervised changepoint detection: This also refers to a method for detecting changepoints in one variable’s behavior, similar to the above. The main difference is that we assume minimal knowledge

  * Self-supervised modelling: This component refers to regression models that capture the target variable’s expected behavior
  * Semi-supervised modelling: This refers to the regression model used in the semi-supervised changepoint detection component that aims to capture the expected behavior of our target variable
  * Deviation detection:  Our models for the expected behavior aim to provide a tool for detecting periods where the target variable has a deviating behavior. This deviating behavior is typically slowly progressing, so it cannot be detected as a changepoint, since changepoints refer to rapid changes.  This component provides a toolkit for analysing historical data and detecting deviating periods in a time series which is completely known when our models are deployed.    
  * Real-time deviation detection: Using our models for the expected behavior of a target variable, we are also able to make real-time deviation detection. Real-time deviation detection refers to a scenario where new data points are received, as a stream.
  
  Link to the notebook: [Deviation Detection](https://github.com/MORE-EU/RES-health/blob/main/notebooks/tools/deviation_detection_basic.ipynb)


2. **Behavior Detection**

 Handling the problem as an event detection task, introducing the requirement to be able to deploy the developed methods in a real-time setting. Namely, our methods need to effectively and efficiently/scalably predict in constantly incoming windows of multidimensional time series, produced by a large number of different turbines in parallel. In our setting, the multidimensional time series consists of a set of variables-measurements on the turbine. The desired event-behavior we aim to detect is the (absolute value of the) static angle between nacelle and wind direction, termed yaw misalignment angle. Given that, we consider two different approaches for modeling and solving the task.
	
1. Directly model: use the yaw misalignment angle as a dependent variable and train regression models that exploit the remaining time series variables as independent variables to predict the former. The regression models are either to be learned on a historical data and be deployed (tested) on the newly incoming data of the same turbine or  to be learned on a time series and deployed on different series. This comprises the most intuitively straightforward approach, allowing to experiment with less variants. 

2. Indirectly model: the yaw misalignment angle by training regression models for approximating the dependent variable and assigning them to different angles. The prediction on newly incoming data is then performed by aggregation of the assigned angles of individual regression models that better approximate the dependent variable on a time window of the new time series. This comprises a more elaborate modelling approach, that allows us to experiment with more variants, e.g., considering aggregation schemes. 


The steps of the Behavior detection tool are presented here.
* Feature Selection: The component which is responsible for the detection of the most important variables.
* Binning: This refers to partitioning the dataset into bins based on the values of an input variable and train specific models on each bins.
* Model Tuning: Tuning the hyper parameters for each model
* Direct Modelling: This component refers to methods modelling the variable that represents the behavior of the time-series at the given point, as the dependent variable. 
* Indirect Modelling: This component includes solutions that work by modelling the behavior of the time-series during (labelled) periods and then assigning labels depending on how well those models approximate the behavior of the newly incoming data. 
* Behavior detection. Refers to the detection of the behavior of newly incoming data based on the available labels and utilizing an aggregate of the predictions of the multiple models trained in different bins or regions of the training datasets. 

  Link to the notebook: [Direct Behavior Detection ](https://github.com/MORE-EU/RES-health/blob/main/notebooks/tools/Direct_model_Evaluation_method.ipynb)
  
  Link to the notebook: [Indirect Behavior Detection ](https://github.com/MORE-EU/RES-health/blob/main/notebooks/tools/Indirect_model_Evaluation_Naive.ipynb)
  
3. **Forecasting**

 This section discusses the challenges and importance of forecasting the power output of wind turbines in the context of variable renewable energy sources (vRES). As vRES penetration levels increase, they introduce more variability and unpredictability to electricity generation, leading to volatile energy market prices. The objective of this tool is to predict a safe underestimate of the minimum power output for each hour over the next two days based on the weather data.
 
 The steps of the Forecasting tool are presented here.
* Data preprocessing: This refers to our initial steps of our algorithm for detecting outliers, normalizing the data and filling out any missing data. Standard techniques like IQR outlier detection and Iterative Imputation are employed throughout these preprocessing steps
* Feature engineering: This refers to introducing features for encoding the time component and lagged features to take into account dependencies between the target (future) variable and recent, past measurements.
*  Model training: This refers to fitting a regression model on a set of historical data that have been preprocessed as described above. For this step, we employ a gradient boosting framework that uses a histogram-based approach to efficiently train decision tree models, namely [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html)
* Prediction: This refers to using the abovementioned model to make predictions of the target variable for the future. To account for multiple future values, we employ a regression chain, a standard machine learning technique used for multi-output regression problems
  
  Link to the notebook: [Forecasting](https://github.com/MORE-EU/RES-health/blob/main/notebooks/tools/Forecasting_tool.ipynb)

4. **Underperformance Detection**

In this section, we present our novel tool for estimating and extracting periods of underperformance in wind data, as well as for detecting underperformance in newly incoming data. Specifically, the problem involves identifying instances in the time series where the target variable exhibits values significantly lower than expected.

The steps of the Underperformance detection tool are presented here.
* Quantile regression: This refers to fitting a quantile regression model for one of the highest quantiles, e.g., 0.8 or 0.9, in a set of historical data. Intuitively, this model aims to approximate the optimal performance (or an overestimation) from a set of data that possibly contains underperforming periods.
* Underperformance extraction: This refers to using the above model to extract periods of underperformance in the same historical data as follows: we compute the (signed) residuals between the predicted and the actual values and we extract periods during which we observe consecutive large residuals, i.e., the actual values are unusually small for a substantial period of time. This step essentially assigns labels to the data, which allows us to employ supervised methods
* Classifier training: Having extracted periods of underperformance, we train a LightGBM classifier to predict whether a certain feature vector is associated with underperformance or not. We use as feature variables the ones used in quantile regression, along with the target variable (whose “underperformance” is to be detected)
* Prediction: Using the abovementioned classifier, we can determine whether new incoming data correspond to underperformance


  Link to the notebook: [Underperformance Detection](https://github.com/MORE-EU/RES-health/blob/main/notebooks/tools/underperformance_detection.ipynb)


5. **Oscillation Detection**
In this section, we present our tool for the detection of oscillations in streams of voltage/current measurement. Within the scope of this use case, we explore automated methods for the real-time detection of oscillations. Detecting oscillations can be a challenging task due to various reasons such as noise in the data. Moreover, power systems can exhibit different types of oscillations such as inter-area oscillations or local oscillations. Our use-case scenario concerns oscillations measured at the Point of Connection (POC). POC is the point that the plant is connected to the grid. Such oscillations may have a frequency of 0.1 to 2.5 Hz.

The steps of Oscillation detection tool are presented here.
* Approximation: This step consists of methods for approximating a signal that is assumed to behave as an oscillation in the sense that it can be accurately approximated as a sinusoidal function (or a sum of sinusoidal functions). We implement various methods including methods that have been previously employed in the context of oscillation analysis (Prony’s method, Matrix-Pencil), non-linear regression by means of the Levenberg-Marquardt algorithm, and a simple Fast Fourier Transformation (FFT) analysis.
* Alerting: Using the abovementioned sinusoidal approximation of a recent window of signal, we calculate the residuals, and by comparing them with the previously seen values we determine whether the newly retrieved data can be approximated by a sinusoidal unusually well.
  
  Link to the notebook: [Oscillation Detection](https://github.com/MORE-EU/RES-health/blob/main/notebooks/tools/OS_comparison_all_methods.ipynb)

6. **Generic Pattern Matching**
   The set of tools for pattern extraction also includes a generic pattern matching algorithm namely Frechet-based Efficient Time-series Comparison Heuristic. The repository can be found in:  [Fetch](https://github.com/MORE-EU/fetch)

Installation
============
Python 3.8.5+ is required. For a guide on how to install python, one can visit [Python](https://www.python.org/). In order to check the version of python installed in the system, one can run:

**$python  -- version**

One should start by downloading the latest version of our source code. If git is installed, this step can be implemented as follows:

**$ git clone https://github.com/MORE-EU/RES-health/tree/main -- recursive**

One new folder under the name RES-health will be created. Let <path> be the absolute path to that folder, i.e., the new folder is <path>/RES-health.

It is preferable to use a virtual environment, which will host all necessary libraries, as follows:

**$ python3 -m venv <path/to/new/virtualenv/>**

**$ source <path/to/new/virtualenv/>/bin/activate**

Now, with the virtual environment activated one can proceed to install the required libraries.

The required libraries are:

• pandas - 1.3.3

• matplotlib - 3.4.2

• numpy - 1.20.1

• scikit_learn - 1.0.2

The above libraries are included in the requirements.txt file. In addition, **jupyter-notebook** must also be installed by executing the following command:

**$ pip install pip install notebook**

We should also note that if one decided to use a virtual environment, they have to install jupyter-notebook, according to the instructions, inside the activated virtual environment.

Now in order to run the notebook implementing this tool, one first needs to start jupyter-notebook (Note, if a virtual environment is used start the notebook inside the activated environment):

**$ jupyter-notebook**

This will open a new tab in the default internet browser where one must select and run the following Jupyter notebook file:

**path/RES-health/notebooks/tools/<notebook>**


Documentation
=============

Source code documentation is available from GitHub pages [Link](https://more-eu.github.io/more-pattern-extraction/)
