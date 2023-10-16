import numpy as np
import pandas as pd
import modules.statistics as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from modules.preprocessing import enumerate2
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import TheilSenRegressor
import sklearn.metrics
from sklearn.metrics import mean_squared_error as mse
from scipy import signal
import math as mt
import scipy.linalg as sl
import scipy.optimize as opt
import statsmodels.api as sm



def predict(df_test, model, feats, target):
    """
    Applies a regression model to predict values of a dependent variable for a given dataframe and 
    given features.

    Args:
        df_test: The input dataframe.
        model: The regression model. Instance of Pipeline.
        feats: List of strings: each string is the name of a column of df_test.
        target: The name of the column of df corresponding to the dependent variable.
    Returns:
        y_pred: Array of predicted values. 
    """

    df_x = df_test[feats]
    df_y = df_test[target]
    X = df_x.values
    y_true = df_y.values
    y_pred = model.predict(X)
    return y_pred


def fit_linear_model(df, feats, target, a=1e-4, deg=3, method='ridge', fit_intercept=True, include_bias=True):
    """
    Fits a regression model on a given dataframe, and returns the model, the predicted values and the associated 
    scores. Applies Ridge Regression with polynomial features. 

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        a: A positive float. Regularization strength parameter for the linear least squares function 
        (the loss function) where regularization is given by the l2-norm. 
        deg: The degree of the regression polynomial.

    Returns:    
        pipeline: The regression model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """

    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values
    polynomial_features = PolynomialFeatures(degree=deg, include_bias=include_bias)
    if method == 'ridge':
        model = Ridge(alpha=a, fit_intercept=fit_intercept)
        
    elif method == 'ols':
        model = LinearRegression(fit_intercept=fit_intercept)
    elif method == 'rf':
        model = RFRegressor(n_jobs = -1)
    else:
        print('Unsupported method')
    

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("regression", model)])
    

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe, med = st.score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe, med


def get_line_and_slope(values):
    """
    Fits a line on the 2-dimensional graph of a regular time series, defined by a sequence of real values. 

    Args:
        values: A list of real values.

    Returns: 
        line: The list of values as predicted by the linear model.
        slope: Slope of the line.
        intercept: Intercept of the line.   
    """

    ols = LinearRegression()
    X = np.arange(len(values)).reshape(-1,1)
    y = values.reshape(-1,1)
    ols.fit(X, y)
    line = ols.predict(X)
    slope = ols.coef_.item()
    intercept = ols.intercept_.item()
    return line, slope, intercept


def train_on_reference_points(df, w_train, ref_points, feats, target, random_state=0):
    """
    Trains a regression model on a training set defined by segments of a dataframe. 
    These segments are defined by a set of starting points and a parameter indicating their duration. 
    In each segment, one subset of points is randomly chosen as the training set and the remaining points 
    define the validation set.
    
    Args:
        df: Input dataframe. 
        w_train: The duration, given as a number of days, of the segments where the model is trained.
        ref_points: A list containing the starting date of each segment where the model is trained.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        random_state: Seed for a random number generator, which is used in randomly selecting the validation 
        set among the points in a fixed segment.

    Returns:
        model: The regression model. This is an instance of Pipeline.
        training_scores: An array containing scores for the training set. It contains the coefficient 
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
        validation_scores: An array containing scores for the validation set. It contains the coefficient 
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
    """

    df_train = pd.DataFrame([])
    df_val = pd.DataFrame([])
    for idx in range(ref_points.size):
        d_train_stop = pd.to_datetime(ref_points[idx]) + pd.Timedelta(days=w_train)
        df_tmp = df.loc[ref_points[idx]:str(d_train_stop)]
        df_tmp2 = df_tmp.sample(frac=1, random_state=random_state) # added random state for reproducibility during experiments
        size_train = int(len(df_tmp2) * 0.80)
        df_train = df_train.append(df_tmp2[:size_train])
        df_val = df_val.append(df_tmp2[size_train:])

    model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train, Me_train = fit_linear_model(df_train, feats, target)
    y_pred_val = predict(df_val, model, feats, target)
    r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val = st.score(df_val[target].values, y_pred_val)
    training_scores = np.array([r_sq_train, mae_train, me_train, mape_train, Me_train])
    validation_scores = np.array([r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val])

    print('Training Metrics:')
    print(f'MAE:{training_scores[1]:.3f} \nME(true-pred):{training_scores[2]:.3f} \nMAPE:{training_scores[3]:.3f} \nR2: {training_scores[0]:.3f}\n')
    print('Validation Metrics:')
    print(f'MAE:{validation_scores[1]:.3f} \nME(true-pred):{validation_scores[2]:.3f} \nMAPE:{validation_scores[3]:.3f} \nMPE:{validation_scores[4]:.3f} \nR2: {validation_scores[0]:.3f}\n')
    return model, training_scores, validation_scores


def predict_on_sliding_windows(df, win_size, step, model, feats, target):
    """
    Given a regression model, predicts values on a sliding window in a dataframe
    and outputs scores, a list of predictions and a list of windows.

    Args:
        df: The input dataframe.
        win_size: The size of the sliding window, as a number of days.
        step: The sliding step.
        model: The regression model.
        feats: A list of names of columns of df indicating the feature variables.
        target: The name of a column of df indicating the dependent variable.

    Returns:
        scores: An array of arrays of scores: one array for each window containing the coefficient of
        determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error,
        the mean percentage error.
        preds_test: a list of predictions: one list of predicted values for each window.
        windows: A list of starting/ending dates: one for each window.
    """

    windows = []
    preds_test = []
    scores_list = []
    for i, time in enumerate2(min(df.index), max(df.index), step=step):
        window = pd.to_datetime(time) + pd.Timedelta(days=win_size)
        df_test = df.loc[time:window]
        if df_test.shape[0]>0:
            y_pred = predict(df_test, model, feats, target)
            r_sq, mae, me, mape, mpe, Me = st.score(df_test[target].values, y_pred)
            scores_list.append([r_sq, mae, me, mape, mpe, Me])
            preds_test.append(y_pred)
            windows.append((time, window))
    scores = np.array(scores_list)
    return scores, preds_test, windows


def predict_on_sliding_windows(df, win_size, step, model, feats, target):
    """
    Given a regression model, predicts values on a sliding window in a dataframe
    and outputs scores, a list of predictions and a list of windows.

    Args:
        df: The input dataframe.
        win_size: The size of the sliding window, as a number of days.
        step: The sliding step.
        model: The regression model.
        feats: A list of names of columns of df indicating the feature variables.
        target: The name of a column of df indicating the dependent variable.

    Returns:
        scores: An array of arrays of scores: one array for each window containing the coefficient of
        determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error,
        the mean percentage error.
        preds_test: a list of predictions: one list of predicted values for each window.
        windows: A list of starting/ending dates: one for each window.
    """

    windows = []
    preds_test = []
    scores_list = []
    for i, time in enumerate2(min(df.index), max(df.index), step=step):
        window = pd.to_datetime(time) + pd.Timedelta(days=win_size)
        df_test = df.loc[time:window]
        if df_test.shape[0]>0:
            y_pred = predict(df_test, model, feats, target)
            r_sq, mae, me, mape, mpe, Me = st.score(df_test[target].values, y_pred)
            scores_list.append([r_sq, mae, me, mape, mpe, Me])
            preds_test.append(y_pred)
            windows.append((time, window))
    scores = np.array(scores_list)
    return scores, preds_test, windows


def fit_pipeline(df, feats, target, pipeline, params):
    """
    Fits a regression pipeline on a given dataframe, and returns the fitted pipline,
    the predicted values and the associated scores.

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        pipeline: A pipeline instance, a scikit-learn object that sequentially applies a list of given 
                  preprocessing steps and fits a selected machine learning model.
        params: A dictionary that contains all the parameters that will be used by `pipeline`.
                The dictionary keys must follow the scikit-learn naming conventions.

    Returns:    
        pipeline: The fitted model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """
    
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values

    pipeline.set_params(**params)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe, _ = st.score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe


def lasso_selection(df, features, target, alphas=None):
    """
    Utilizes Lasso regression, which penalizes the l1 norm of the weights and indtroduces sparsity in the solution, 
    to find the most `relevant` features, i.e. the ones that have non-zero weights

    Args:
        df: DataFrame that contains the dataset.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        alphas: A list of regularization coefficients to be used, if left as None the alphas are set automatically.

    Returns: 
        selected_features: A list with the name of the columns that were selected by the Lasso method.
    """
    
    X = df[features]
    y = df[target]
    lasso = LassoCV(cv=5, random_state=42, alphas=alphas).fit(X, y)
    model = SelectFromModel(estimator=lasso, prefit=True)
    sup = model.get_support()
    selected_features = list(np.array(features)[sup])
    return selected_features


def perform_grid_search(df, feats, target, scorer, model, params, randomized=False):
    
    """
    Performs a grid-search to find the parameters in the provided search space that yield the best results.
    Used for model tuning.

    Args:
        df: DataFrame that contains the dataset.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        scorer: A performance metric or loss function used in the grid search.
        model: The model that will be tuned.
        params: A dictionary containing all the parameters to be tested.
        randomized: If set to True, a random sample of all the parameter combinations will be tested.

    Returns: 
        selected_params: Dictionary that contains the combination of parameters that resulted in the best score during grid search.
    """
    
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values
    
    model = model
    
    if randomized == True:
        grid_pipeline = RandomizedSearchCV(model, params, n_iter=100, verbose=0, n_jobs=-1, pre_dispatch=64, scoring=scorer, cv=3, random_state=42).fit(X, y)
    else:
        grid_pipeline = GridSearchCV(model, params, verbose=0, n_jobs=-1, pre_dispatch=64, scoring=scorer, cv=3).fit(X, y)
   
    selected_params = {}
    for key in  params:
        selected_params[key] = grid_pipeline.best_params_[key]
    
    return selected_params


def get_ts_line_and_slope(values):
    """
    Fits a line on the 2-dimensional graph of a regular time series, defined by a sequence of real values. 

    Args:
        values: A list of real values.

    Returns: 
        line: The list of values as predicted by the linear model.
        slope: Slope of the line.
        intercept: Intercept of the line.   
    """

    ols = TheilSenRegressor(random_state=0)
    X = np.arange(len(values)).reshape(-1,1)
    y = values.reshape(-1,1)
    
    ols.fit(X, y.ravel())
    line = ols.predict(X)
    slope = ols.coef_.item()
    intercept = ols.intercept_.item()
    return line, slope, intercept  


def calc_changepoints_one_model(df, dates_rain_start, dates_rain_stop, model, target, feats, w1, w2):
    """
    Returns errors associated with changepoint detection in the input segments. Applies the method using one 
    model in all segments (Method 2).
    Args:
        df: Input pandas dataframe
        dates_rain_start: Array of starting points of segments under investigation 
        dates_rain_stop: Array of ending points of segments under investigation 
        model: Regression model 
        target: Name of dependant variable in model
        feats: List of feature variables in model
        w1: Number of days defining the period before each segment, which will be used for calculating the associated score 
        w2: Number of days defining the period after each segment, which will be used for calculating the associated score 
    Returns:
        errors_br: Array containing prediction errors before each segment  
        errors_ar: Array containing prediction errors after each segment 
    """
    errors_br = np.empty((dates_rain_start.size, 6))
    errors_ar = np.empty((dates_rain_start.size, 6))
    for i in range(dates_rain_start.size):
        d1 = dates_rain_start.iloc[i]
        d0 = d1 - pd.Timedelta(days=w1)
        d2 = dates_rain_stop.iloc[i]
        d3 = d2 + pd.Timedelta(days=w2)
        df_ar = df[d2:d3]
        df_br = df[d0:d1]
        try:
            y_pred_ar = predict(df_ar, model, feats, target)
            y_pred_br = predict(df_br, model, feats, target)
            errors_ar[i,:] = st.score(df_ar[target].array, y_pred_ar)
            errors_br[i,:] = st.score(df_br[target].array, y_pred_br)
        except:
            errors_ar[i,:] = [np.nan]*6
            errors_br[i,:] = [np.nan]*6
    return errors_br, errors_ar        


def calc_changepoints_many_models(df, dates_rain_start, dates_rain_stop, target, feats, w1, w2, w3):
    """
    Returns errors associated with changepoint detection in the input segments. Applies the method using one 
    model for each segment (Method 1).
    Args:
        df: Input pandas dataframe
        dates_rain_start: Array of starting points of segments under investigation 
        dates_rain_stop: Array of ending points of segments under investigation 
        target: Name of dependant variable in model
        feats: List of feature variables in model
        w1: Number of days defining the period before each segment, which will be used for training the model
        w2: Number of days defining the period before each segment, which will be used for calculating the associated score 
        w3: Number of days defining the period after each segment, which will be used for calculating the associated score 
    Returns:
        errors_br: Array containing prediction errors before each segment  
        errors_ar: Array containing prediction errors after each segment 
    """
    errors_br = np.empty((dates_rain_start.size, 6))
    errors_ar = np.empty((dates_rain_start.size, 6))
    for i in range(dates_rain_start.size):
        d1 = dates_rain_start.iloc[i]
        d2 = dates_rain_stop.iloc[i]
        try:
            y_pred_train, score_train, y_pred_val, errors_br[i,:], y_pred_test, errors_ar[i,:] = changepoint_scores(df, feats, target, d1, d2, w1, w2, w3)
        except:
            errors_ar[i,:] = [np.nan]*6
            errors_br[i,:] = [np.nan]*6
    return errors_br, errors_ar 
def changepoint_scores(df, feats, target, d1, d2, w_train, w_val, w_test):
    """
    Given as input a dataframe and a reference interval where a changepoint may lie, trains a regression model in
    a window before the reference interval, validates the model in a window before the reference interval and tests 
    the model in a window after the reference interval. 

    Args:
        df: The input dataframe.
        feats: A list of names of columns of df indicating the feature variables.
        target: The name of a column of df indicating the dependent variable.
        d1: The first date in the reference interval.
        d2: The last date in the reference interval.
        w_train: The number of days defining the training set.
        w_val: The number of days defining the validation set.
        w_test: The number of days defining the test set.
    Returns:
        y_pred_train: The array of predicted values in the training set.
        score_train: An array containing scores for the training set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
        y_pred_val: The array of predicted values in the validation set.
        score_val: An array containing scores for the validation set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
        y_pred_test: The array of predicted values in the test set.
        score_test: An array containing scores for the test set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
    """

    d_train_start = pd.to_datetime(d1) - pd.Timedelta(days=w_train) - pd.Timedelta(days=w_val)
    d_train_stop = pd.to_datetime(d1) - pd.Timedelta(days=w_val)
    d_test_stop = pd.to_datetime(d2) + pd.Timedelta(days=w_test)
    df_train = df.loc[str(d_train_start):str(d_train_stop)]
    df_val = df.loc[str(d_train_stop):str(d1)]
    df_test = df.loc[str(d2):str(d_test_stop)]
    if len(df_train) > 0 and len(df_test) > 0:
        model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train, Me_train = fit_linear_model(df_train, feats, target)
        y_pred_val = predict(df_val, model, feats, target)
        y_pred_test = predict(df_test, model, feats, target)
        
        r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val = st.score(df_val[target].values, y_pred_val)
        r_sq_test, mae_test, me_test, mape_test, mpe_test, Me_test = st.score(df_test[target].values, y_pred_test)
        score_train = np.array([-r_sq_train, mae_train, me_train, mape_train, mpe_train, Me_train])
        score_val = np.array([-r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val])
        score_test = np.array([-r_sq_test, mae_test, me_test, mape_test, mpe_test, Me_test])
        return y_pred_train, score_train, y_pred_val, score_val, y_pred_test, score_test
    else:
        raise Exception("Either the training set is empty or the test set is empty")
        
        
        
def procedure(df,df_soil_output,pop_size,days,
              num_generations,num_parents_mating,num_mutations,
              col,events,parenting,crossover,mix_up=True):
    """
    The whole genetic algorithm procedure. Returns best outputs of fitness,
    the last survived population, end_df: the frame created by all iterations,
    alles_df: the last df with the best results
    
    Args:
        df (pandas.DataFrame): The main dataframe.
        df_soil_output (pandas.DataFrame): DataFrame of soiling events.
        pop_size (int): The population size.
        days (int): Shift in the index of soiling events by days.
        num_generations (int): Number of generations.
        num_parents_mating (int): Number of parents to mate.
        num_mutations (int): Number of mutations to apply.
        col (list): Columns of the dataframe to perform the routine.
        events (int): Number of soiling periods.
        parenting (str): Parent selection method ('smp', 'sss', 'ranks', 'randoms', 'tournament', 'rws', 'sus').
        crossover (str): Crossover method ('single', 'twopoint', 'uni', 'scatter', 'old').
        mix_up (bool): Whether to mix up parenting and crossover methods.
    
    Returns:
        numpy.ndarray: Last survived population.
        list: Best fitness scores for each generation.
        pandas.DataFrame: DataFrame created by all iterations.
        pandas.DataFrame: Last DataFrame with the best results.
    """
  #    print(f'pop_size:{pop_size}')
#     print(f'num_gen:{num_generations}')

    #Creating the initial population.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import sklearn.metrics
    start = time.time()

    new_population=initilization_of_population_mp(pop_size,events)
    # print(new_population)
    # print(f'new_population:{new_population}')
    # Number of the weights we are looking to optimize.
    num_weights = len(new_population[0,:])
    print(f'Features: {col}')
    print(f'Chromosomes: {len(new_population[0,:])}')
    print(f'Soiling Events: {events}')
    print(f'Generations: {num_generations}')
    print(f'Population :{len(new_population)}')
    print(f'Parents: {num_parents_mating}')
    
#     print(f'num_weights: {num_weights}')
    best_outputs = []
    end_df=pd.DataFrame()
    for generation in tqdm(range(num_generations)):
        # Measuring the fitness of each chromosome in the population.
        fitness,alles_df = fiteness_fun(df,df_soil_output,days,new_population,col)
        result = [] 
        for i in fitness: 
            if i not in result: 
                result.append(i)
            else: 
                result.append(0)
        fitness=result
#         print(generation,np.max(fitness))
#         print(alles_df.head(1))
        # Thei best result in the current iteration.  
#         print(np.max(fitness))
        if mix_up:
            parenting=random.choice(['sss','ranks','randoms','tournament','rws'])
#         print(parenting)
        best_outputs.append(np.max(fitness))
        # Selecting the best parents in the population for mating.
        if parenting=='smp':
            parents = select_mating_pool(new_population, fitness, 
                                          num_parents_mating)
        elif parenting=='sss':
            parents = steady_state_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='ranks':
            parents = rank_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='randoms':
            parents = random_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='tournament':
            parents = tournament_selection(new_population,fitness, num_parents_mating,toursize=100)[0]
        elif parenting=='rws':
            parents = roulette_wheel_selection(new_population,fitness, num_parents_mating)[0]
        elif parenting=='sus':
            parents = stochastic_universal_selection(new_population,fitness, num_parents_mating)[0]
        else:
            raise TypeError('Undefined parent selection type')
        # Generating next generation using crossover.
        offspring_size=(len(new_population)-len(parents), num_weights)
        if mix_up:
            crossover=random.choice(['single','twopoint','uni','scatter'])

#         print(crossover)




        if crossover=='single':
            offspring_crossover=single_point_crossover(parents, offspring_size,crossover_probability=None)
        elif crossover=='twopoint':
            offspring_crossover=two_points_crossover(parents, offspring_size,crossover_probability=None)
        elif crossover=='uni':
            offspring_crossover=uniform_crossover(parents, offspring_size,crossover_probability=None)
        elif crossover=='scatter':
            offspring_crossover=scattered_crossover(parents, offspring_size,crossover_probability=None,num_genes=num_weights)
        elif crossover=='old':
            offspring_crossover = crossover(parents,
                                           offspring_size=(len(new_population)-len(parents), num_weights))
        else:
            raise TypeError('Undefined crossover selection type')

        
        # Adding some variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, num_mutations)
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        end_df=pd.concat([alles_df,end_df],axis=0)

    end = time.time()
    print(f'Time to complete: {np.round(end - start,2)}seconds')
    return new_population,best_outputs,end_df,alles_df




def fiteness_fun(df,df_soil_output,days,new_population,col):
     """
    Calculate fitness values for each individual in the population.
    
    Args:
        df (pandas.DataFrame): The main dataframe.
        df_soil_output (pandas.DataFrame): DataFrame of soiling events.
        days (int): Shift in the index of soiling events by days.
        new_population (numpy.ndarray): The population of individuals with attributes.
        col (list): Columns of the dataframe to perform the routine.
        
    Returns:
        list: Fitness values for each individual.
        pandas.DataFrame: DataFrame containing detailed information about each individual's performance.
    """
    fitness_each_itter=[]
    list_score_jac=[]
    alles_df=pd.DataFrame()
    
    for row in (range(len(new_population))):

        if new_population[:,5][row]<3:
            new_population[:,5][row]=3
        if new_population[:,4][row]<1:
            new_population[:,4][row]=1
            
       
            
        conc=pd.DataFrame()
        md,mi,excluzion_zone=pmc(df=df,new_population=new_population,row=row,col=col)
        lista_jac_mtype=[]
        for n,k in enumerate(range(len(mi))):
            test=tester(df,new_population,row,df_soil_output,mi,k)
            f1,ps,recal,hamming,jaccard,cohen,roc=scorer(test)
            lista_jac_mtype.append(f1)
            data={'min_nei':int(new_population[row,0]),
                 'max_d':new_population[row,1],
                 'cutoff':new_population[row,2],
                 'max_m':int(new_population[row,3]),
                 'max_motif':int(new_population[row,4]),
                 'profile_wind':int(new_population[row,5]),
                 'exclusion_zone':excluzion_zone,
                 'motif_type': n+1,
                 'actual_nei':len(mi[k]),
                 'actual_motif':len(md),
                 'recall':np.round(recal,5),
                 'f1':np.round(f1,5),
                  'precision':np.round(ps,5),
                  'hamming':np.round(hamming,5),
                  'jaccard':np.round(jaccard,5),
                 'cohen':np.round(cohen,5),
                 'roc':np.round(roc,5)}
            
             
            conc=pd.concat([conc,pd.DataFrame(data,index=[0])],axis=0)
        alles_df=pd.concat([conc,alles_df],axis=0)
        list_score_jac.append(np.max(lista_jac_mtype))
    alles_df=alles_df.loc[alles_df[['f1']].drop_duplicates(['f1']).index]
    alles_df=alles_df.sort_values(by=['f1'], ascending=False).reset_index(drop=True)
    return list_score_jac,alles_df

    


def clean_motifs(md,mi):
    """
    Clean found motifs from trivial motifs or dirty neighbors.
    
    Args:
        md (list): Motif distance.
        mi (list): Motif indexes.
        
    Returns:
        list: Cleaned motif distances.
        list: Cleaned motif indexes.
    """
    outp=[]
    for j in range(0,len(mi)):
        outp.append(np.delete(mi[j], np.where(mi[j] == -1)))
    mi=outp    
    outp=[]
    for j in range(0,len(md)):
        outp.append(md[j][~np.isnan(md[j])])
    md=outp
    return md,mi

def pmc(df,new_population,row,col):


    """
    pmc: profile, motif, cleaning
    Creates a pipeline calculating the profile, the motifs, and cleaning them for each individual.
    
    Args:
        df (pandas.DataFrame): The main dataframe.
        new_population (numpy.ndarray): Population of individuals.
        row (int): The index of each individual.
        col (list): Columns of the dataframe to perform the routine.
        
    Returns:
        list: Motif distances.
        list: Motif indexes.
        float: Exclusion zone denominator.
    """

    from stumpy import mstump
    from stumpy import mmotifs
    x=random.choice([np.inf,1,2,3,4,5,6,7,8])
    stumpy.config.STUMPY_EXCL_ZONE_DENOM = x

    mp,mpi=stumpy.mstump(df[col].to_numpy().transpose(), m=int(new_population[row][5]),discords=False,normalize=True)

    md,mi,sub,mdl=stumpy.mmotifs(df[col].to_numpy().transpose(),mp,mpi,
                             min_neighbors=int(new_population[row][0]),
                             max_distance=new_population[row][1],cutoffs=new_population[row][2],
                             max_matches=int(new_population[row][3]),max_motifs=int(new_population[row][4]))  
#     print(stumpy.config.STUMPY_EXCL_ZONE_DENOM)
    md,mi=clean_motifs(md,mi)
    return md,mi,stumpy.config.STUMPY_EXCL_ZONE_DENOM


def tester(df,new_population,row,df_soil_output,mi,k):
     """
    Tester function to generate predictions for a given motif.
    
    Args:
        df (pandas.DataFrame): The main dataframe.
        new_population (numpy.ndarray): Population of individuals.
        row (int): The index of the individual.
        df_soil_output (pandas.DataFrame): DataFrame of soiling events.
        mi (list): Motif indexes.
        k (int): Index of the motif to test.
        
    Returns:
        pandas.DataFrame: DataFrame with predicted and actual values.
    """
    test=pd.DataFrame()
    test.index=df.index
    test['pred']=np.nan
    for i in mi[k]:
        test['pred'].iloc[i:i+int(new_population[row,5])]=1
    test['pred'] = test['pred'].fillna(0)
    test["actual"] = np.nan
    for start, end in zip(df_soil_output.SoilStart, df_soil_output.SoilStop):
        test.loc[start:end, 'actual'] = 1
    test['actual'] = test['actual'].fillna(0)
    return test



def naive_baseline(df_test):
    """
    Generate a naive baseline prediction for a test DataFrame.

    Args:
        df_test (pandas.DataFrame): Test DataFrame.

    Returns:
        numpy.ndarray: Naive baseline predictions.
    """
    preds = []
    for i in range(df_test.shape[0]):
        t = [x for x in df_test.columns if 'Grd_Prod_Pwr_min_(t+' in x]
        pred = np.zeros(len(t))
        pred = pred + df_test.iloc[i, :]['Grd_Prod_Pwr_min']
        preds.append(pred)
    preds = np.vstack(preds)
    return preds




def estimate_freq(series):
    """
    Estimate a principal frequency component using FFT.

    Parameters:
        series (array-like): The input signal to estimate.

    Returns:
        freq: estimated frequency
    """
    # Nearest size with power of 2
    size = 2 ** np.ceil(np.log2(2*len(series) - 1)).astype('int')
    # Variance
    var = np.var(series)
    # Normalized data
    ndata = series - np.mean(series)
    # Compute the FFT
    fft = np.fft.fft(ndata, size)
    # Get the power spectrum
    pwr = np.abs(fft) ** 2
    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = np.fft.ifft(pwr).real / var / len(series)
    peaks = signal.find_peaks(acorr[:len(series)])
    #### does it take the highest?
    if len(peaks[0]) == 0:
        return np.nan
    else:
        peak_idx = peaks[0][0]
        freq = 1/(series.index[peak_idx]-series.index[0]).total_seconds()
        return freq

def fit_sin_simple(df):
    """
    Estimate a signal using an FFT-based method.

    Parameters:
        df (array-like): The input signal to estimate.
        

    Returns:
        A tuple (data_fit, y_norm) containing the estimated (normalized) signal and the normalized signal.
    """
    y = df.values
    index = (df.index - min(df.index)).total_seconds()
    var = np.var(y)
    # Normalized data
    y_norm = (y - np.mean(y))/var
    acorr = sm.tsa.acf(y_norm, nlags = len(y_norm)-1)
    peaks = signal.find_peaks(acorr[:len(y_norm)])
    try:
        peak_idx = peaks[0][0]
        est_freq = 1/(df.index[peak_idx]-df.index[0]).total_seconds()
    except:
        est_freq = 1
    
    #find amplitude
    optimize_func = lambda x: (x[0]*np.sin(2*np.pi*index) - y_norm)
    #find phase
    yf = np.fft.fft(y_norm)
    T = 1/30
    freq = np.fft.fftfreq(y.size, d=T)
    ind, = np.where(np.isclose(freq, est_freq, atol=1/(T*len(y))))
    est_phase = np.angle(yf[ind[0]])
    est_amp = np.sqrt(2)*np.std(y_norm)
    data_fit = est_amp*np.sin(2*np.pi*est_freq*index+est_phase) 
    return data_fit, y_norm
    
def fit_sin_lsm(df):
    """
    Estimate a signal using the non-linear least squared method.

    Parameters:
        df (array-like): The input signal to estimate.
        

    Returns:
        A tuple (data_fit, y_norm) containing the estimated (normalized) signal and the normalized signal.
    """
    y = df.values
    index = (df.index - min(df.index)).total_seconds()
    var = np.var(y)
    # Normalized data
    y_norm = (y - np.mean(y))/var
    guess_mean = np.mean(y_norm)
    acorr = sm.tsa.acf(y_norm, nlags = len(y_norm)-1)
    peaks = signal.find_peaks(acorr[:len(y_norm)])
    try:
        peak_idx = peaks[0][0]
        guess_freq = 1/(df.index[peak_idx]-df.index[0]).total_seconds()
    except:
        guess_freq = 1
    #guess_amp = np.mean(y_norm)
    guess_amp = np.sqrt(2)*np.std(y_norm)
 
    #find amplitude
    optimize_func = lambda x: (x[0]*np.sin(2*np.pi*index) - y_norm)

    #find phase     
    yf = np.fft.fft(y_norm)
    T = 1/30
    freq = np.fft.fftfreq(y.size, d=T)
    try:
        ind, = np.where(np.isclose(freq, guess_freq, atol=1/(T*len(y))))
        guess_phase = np.angle(yf[ind[0]])
    except:
        guess_phase = 0
    
    
    #optimize_func = lambda x: np.abs(x[0]*np.sin(2*np.pi*x[1]*index+x[2]) - y_norm)
    #est_amp, est_freq, est_phase = opt.leastsq(optimize_func, [guess_amp, guess_freq, guess_phase], col_deriv = 1)[0]
    est_freq = guess_freq
    optimize_func = lambda x: np.abs(x[0]*np.sin(2*np.pi*index+x[1]) - y_norm)
    est_amp, est_phase = opt.leastsq(optimize_func, [guess_amp, guess_phase], col_deriv = 1)[0]
    data_fit = est_amp*np.sin(2*np.pi*est_freq*index+est_phase) 
    return data_fit, y_norm    


def prony_method(signal, p):
    """
    Estimate a signal using the Prony method.

    Parameters:
        signal (array-like): The input signal to estimate.
        p (int): The number of poles to use for the Prony method.

    Returns:
        A tuple (signal_hat, poles) containing the estimated signal and the estimated poles.
    """
    N = len(signal)

    # Construct the Hankel matrix
    H = np.zeros((N-p, p))
    for i in range(N-p):
        for j in range(p):
            H[i, p-1-j] = signal[i+j]
            
    
    # Solve the Prony system of equations
    b = -signal[p:N]
    
    #a = np.linalg.solve(H, b)
    a = np.linalg.lstsq(H, b)[0]
    
    a = np.concatenate([[1],a])
    # Compute the estimated poles
    z = np.roots(a)
    Z = np.zeros([p,p],dtype=complex)
   
    for i in range(p):
        for j in range(p):
            Z[i,j]=z[j]**i
            
    
    #h=np.linalg.solve(Z,signal[:p])
    h = np.linalg.lstsq(Z,signal[:p])[0]
    
    # Compute the estimated signal
    signal_hat = np.zeros(N)
    for i in range(N):
        signal_hat[i] = np.real(np.sum(h*(z**i)))

    return signal_hat, z


#following is based on https://github.com/Tan-0321/PronyMethod/blob/main/PronyMethod.py
def MPM(phi,p):    
    """
    Estimate a signal using the Matrix Pencil method.

    Parameters:
        phi (array-like): The input signal to estimate.
        p (int): The pencil parameter.

    Returns:
        signal_hat containing the estimated signal.
    """
    end=len(phi)-1
    Y=sl.hankel(phi[:end-p],phi[end-p:])
    Y1=Y[:,:-1]
    Y2=Y[:,1:]
    Y1p=np.linalg.pinv(Y1)
    EV=np.linalg.eigvals(np.dot(Y1p, Y2))
    EL=len(EV)
    
    #complex residues (amplitudes and angle phases)as Prony's method
    
    Z=np.zeros([EL,EL],dtype=complex)
    rZ=np.empty([EL,EL])
    iZ=np.empty([EL,EL])
    for i in range(EL):
        for j in range(EL):
            Z[i,j]=EV[j]**i
            rZ[i,j]=Z[i,j].real
            iZ[i,j]=Z[i,j].imag
    
    h=np.linalg.solve(Z,phi[0:EL])
    theta=np.empty([EL])
    amp=abs(h)
    for i in range(EL):
        theta[i]=mt.atan2(h[i].imag, h[i].real)
        
    # Compute the estimated signal
    signal_hat = np.zeros(len(phi))
    for i in range(len(phi)):
        signal_hat[i] = np.real(np.sum(h*(EV**i)))
        
    #answer=np.c_[amp,theta,alpha,frequency]
    return signal_hat


def fit_pipeline_lgbm(df, feats, target, pipeline, params, init_model=None):
    
    """
    Fits a regression pipeline on a given dataframe, and returns the fitted pipeline,
    the predicted values and the associated scores.

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        pipeline: A pipeline instance, a scikit-learn object that sequentially applies a list of given 
                  preprocessing steps and fits a selected machine learning model.
        params: A dictionary that contains all the parameters that will be used by `pipeline`.
                The dictionary keys must follow the scikit-learn naming conventions.
        init_model: An initial model to be used for boosting.

    Returns:    
        pipeline: The fitted model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """
    import sklearn.metrics as st

    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values

    pipeline.set_params(**params)
    weights = np.linspace(0, 1, df.shape[0])
    weights = weights * weights
    pipeline.fit(X, y, regression__sample_weight=weights, regression__init_model=init_model)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe, _ = st.score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe

def select_params_from_res_MAE(df):
    """
    Select the parameters with the best cross-validation (CV) train-test difference in Mean Absolute Error (MAE).
    Allow up to 10% deviation from the best MAE.
    
    Args:
        df (pandas.DataFrame): DataFrame containing results of grid search with MAE metrics.
        
    Returns:
        dict: Selected parameters.
        str: Model name.
        str: Feature selector name.
    """
    # select the params with the best cv train-test difference
    # allowing for up to 10% deviation from the best MAE
    # note that score numbers from grid search are negative
    df['differ'] = np.abs(df.mean_test_MAE - df.mean_train_MAE)
    best = df.mean_test_MAE.max()
    df = df[df.mean_test_MAE >= best + 0.1 * best]
    df = df.sort_values(by='differ')
    s = df.iloc[0, :]
    params = s['params']
    model_name = s['model']
    selector_name = s['feat_selector']
    return params, model_name, selector_name
