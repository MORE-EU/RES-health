import pandas as pd
import numpy as np
from matrixprofile import core
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize as norm
from sklearn.neighbors import LocalOutlierFactor

def enumerate2(start, end, step=1):
    """ 
    Args:
        start: starting point
        end: ending point    .
        step: step of the process                   
        
        
    Return: 
        The interpolated DataFrame/TimeSeries
     """
    i=0
    while start < pd.to_datetime(end):
        yield (i, start)
        start = pd.to_datetime(start) + pd.Timedelta(days=step)
        i += 1

def change_granularity(df,granularity='30s',size=10**7,chunk=True): 
    """ 
    Changing the offset of a TimeSeries. 
    We do this procedure by using chunk_interpolate. 
    We divide our TimeSeries into pieces in order to interpolate them.
        
    Args:
        df: Date/Time DataFrame. 
        size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.       .
        granularity: The offset user wants to resample the Time Series                  
        chunk: If set True, It applies the chunk_interpolation
    
    Return: 
        The interpolated DataFrame/TimeSeries
     """

    df = df.resample(granularity).mean()
    print('Resample Complete')
    if chunk==True: #Getting rid of NaN occurances.
        df=chunk_interpolate(df,size=size,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1)
        print('Interpolate Complete')
    return df
  
def filter_col(df, col, less_than=None, bigger_than=None): 
    """ 
    Remove rows of the dataframe that they are under, over/both from a specific/two different input price/prices.
        
    Args:
        df: Date/Time DataFrame. 
        col: The desired column to work on our DataFrame. 
        less_than: Filtering the column dropping values below that price.
        bigger_than: Filtering the column dropping values above that price.
    
    Return: 
        The Filtrated TimeSeries/DataFrame
    """
    if(less_than is not None):
        df=df.drop(df[df.iloc[:,col] < less_than].index)
    if(bigger_than is not None):
        df=df.drop(df[df.iloc[:,col] > bigger_than].index)
    print('Filter Complete')
    return df


def filter_dates(df, start, end):
    """ 
    Remove rows of the dataframe that are not in the [start, end] interval.
    
    Args:
        df:DataFrame that has a datetime index.
        start: Date that signifies the start of the interval.
        end: Date that signifies the end of the interval.
   
   Returns:
        The Filtrared TimeSeries/DataFrame
    """
    date_range = (df.index >= start) & (df.index <= end)
    df = df[date_range]
    return df

def normalize(df):
    """ 
    This function transforms an input dataframe by rescaling values to the range [0,1]. 
    
    Args:
        df: Date/Time DataFrame or any DataFrame given with a specific column to Normalize. 
   
    Return:
        Normalized Array
    """
    values=[]
        # prepare data for normalization
    values = df.values
    values = values.reshape((len(values), 1))
        # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    return normalized


def add_noise_to_series(series, noise_max=0.00009):
    
    """ 
    Add uniform noise to series.
    
    Args:
        series: The time series to be added noise.
        noise_max: The upper limit of the amount of noise that can be added to a time series point
    
    Return: 
        DataFrame with noise
    """
    
    if not core.is_array_like(series):
        raise ValueError('series is not array like!')

    temp = np.copy(core.to_np_array(series))
    noise = np.random.uniform(0, noise_max, size=len(temp))
    temp = temp + noise

    return temp


def add_noise_to_series_md(df, noise_max=0.00009):
    
    """ 
    Add uniform noise to a multidimensional time series that is given as a pandas DataFrame.
    
    Args:
        df: The DataFrame that contains the multidimensional time series.
        noise_max: The upper limit of the amount of noise that can be added to a time series point.
   
    Return:
        The DataFrame with noise to all the columns
    """
    
    for col in df.columns:
        df[col] = add_noise_to_series(df[col].values, noise_max)
    return df


def filter_df(df, filter_dict):
    """ 
    Creates a filtered DataFrame with multiple columns.
        
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        filter_dict: A dictionary of columns user wants to filter
    
    Return: 
        Filtered DataFrame
    """

    mask = np.ones(df.shape[0]).astype(bool)
    for name, item in filter_dict.items():
        val, perc = item
        if val is not None:
            mask = mask & (np.abs(df[name].values - val) < val * perc)
            
    df.loc[~mask, df.columns != df.index] = np.NaN
    f_df = df
    print(f_df.shape)
    return f_df

def chunker(seq, size):
    """
    Dividing a file/DataFrame etc into pieces for better hadling of RAM. 
    
    Args:
        seq: Sequence, Folder, Date/Time DataFrame or any Given DataFrame.
        size: The size/chunks we want to divide our Seq/Folder/DataFrame.
    
    Return:
        The divided groups
        
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def chunk_interpolate(df,size=10**6,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1):

    """
    After Chunker makes the pieces according to index, we Interpolate them with *args* of pandas.interpolate() and then we Merge them back together.
    This step is crucial for the complete data interpolation without RAM problems especially in large DataSets.
    
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.
    
    Return:
        The Interpolated DataFrame
    """
    
    group=[]
    for g in chunker(df,size):
        group.append(g)
    print('Groupping Complete')
    for i in range(len(group)):
            group[i].interpolate(method=method,axis=axis,limit_direction = limit_direction, limit = limit, inplace=True)
            df_int=pd.concat(group[i] for i in range(len(group)))
            df_int=pd.concat(group[i] for i in range(len(group)))
    print('Chunk Interpolate Done')
    return df_int


def is_stable(*args, epsilon):
    """
    Args:
        epsilon: A small value in order to avoid dividing with Zero.
    
    Return: 
        A boolean vector from the division of variance with mean of a column.
    """
    #implemented using the index of dispersion (or Fano factor)
    dis = np.var(np.array(args),axis = 1)/np.mean(np.array(args),axis = 1)
    return np.all(np.logical_or((dis < epsilon),np.isnan(dis)))


def filter_dispersed(df, window, eps):
    """
    We are looking at windows of consecutive row and calculate the mean and variance. For each window if the index of disperse or given column is in the given threshhold
    then the last row will remain in the data frame.
    
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        window: A small value in order to avoid dividing with Zero.
        eps: A small value in order to avoid dividing with Zero (See is_stable)
    
    Return: The Filtered DataFrame
    """
    df_tmp = df[rolling_apply(is_stable, window, *df.transpose().values, epsilon= eps)]
    return df_tmp[window:]
  
def scale_df(df):
    """ 
    Scale each column of a dataframe to the [0, 1] range performing the min max scaling
    
    Args:
        df: The DataFrame to be scaled.
    
    Return: Scaled DataFrame
    """
    min_max_scaler = MinMaxScaler()
    df[df.columns] = min_max_scaler.fit_transform(df)
    return df

def standardize_df(df):
    """ 
    Scale each column of a dataframe to the [0, 1] range performing the min max scaling
    
    Args:
        df: The DataFrame to be scaled.
    
    Return: Scaled DataFrame
    """
    standard_scaler = StandardScaler()
    df[df.columns] = standard_scaler.fit_transform(df)
    return df

def unit_norm_df(df):
    """ 
    Scale each column of a dataframe to the [0, 1] range performing the min max scaling
    
    Args:
        df: The DataFrame to be scaled.
    
    Return: Scaled DataFrame
    """
    df[df.columns] = norm(df, axis=0)
    return df


def outliers_IQR(df):
    """
    Remove outliers from a DataFrame using the IQR (Interquartile Range) method.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with outliers removed.
    """
    Q1 = df.quantile(0.10)
    Q3 = df.quantile(0.90)
    IQR = Q3 - Q1
    df_iqr = df[~((df < (Q1 - 1.5 * IQR)) | (df >(Q3 + 1.5 * IQR))).any(axis=1)]
    return df_iqr

def outliers_LoF(df, n_neighbors=300):
     """
    Remove outliers from a DataFrame using the Local Outlier Factor (LoF) method.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        n_neighbors (int): The number of neighbors to consider for the LoF calculation.

    Returns:
        pandas.DataFrame: The DataFrame with outliers removed.
    """
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, n_jobs=16)
    res = clf.fit_predict(df)
    df = df[res == 1]
    return df


def split_to_bins(df, bin_size, mini, maxi, feat):
    """
    Split a DataFrame into bins based on a specified feature.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        bin_size (float): The size of each bin.
        mini (float): The minimum value for binning.
        maxi (float): The maximum value for binning.
        feat (str): The feature column for binning.

    Returns:
        list: List of masks indicating bin membership for each bin.
    """
    bins = np.arange(mini, maxi, bin_size)
    bins = np.append(bins, maxi)
    bin_masks = []
    bin_feature = feat
    for i in range(len(bins) - 1):
        mask = (df[bin_feature]>= bins[i]) & (df[bin_feature] < bins[i + 1])
        bin_masks.append(mask)
    return bin_masks


def create_scaler(dfs):
    """
    Create a scaler for data transformation.

    Args:
        dfs (list of pandas.DataFrames): List of DataFrames to create the scaler.

    Returns:
        sklearn.preprocessing.MinMaxScaler: Fitted MinMaxScaler for data transformation.
    """
    df = pd.concat(dfs)
    min_max_scaler = MinMaxScaler()
    fitted_scaler = min_max_scaler.fit(df)
    return fitted_scaler

def soiling_dates(df,y=0.992,plot=True):
     """
    Extracts soiling event dates from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame with a 'soiling_derate' column.
        y (float): The depth threshold for considering soiling periods.
        plot (bool): Whether to plot the derate with soiling event indications.

    Returns:
        pandas.DataFrame: A DataFrame containing the start and stop dates of soiling events.
    """
    soil = pd.concat([pd.Series({f'{df.index[0]}': 1}),df.soiling_derate])
    soil.index = pd.to_datetime(soil.index)
    df_dates = pd.DataFrame(index = soil.index)
    df_dates["soil_start"] = soil[(soil == 1) & (soil.shift(-1) < 1)] # compare current to next
    df_dates["soil_stop"] = soil[(soil == 1) & (soil.shift(1) < 1)] # compare current to prev
    dates_soil_start = pd.Series(df_dates.soil_start.index[df_dates.soil_start.notna()])
    dates_soil_stop = pd.Series(df_dates.soil_stop.index[df_dates.soil_stop.notna()])

    #Filter significant rains with more than 'x' percipitation
    ids = []
    x=y
    for idx in range(dates_soil_start.size):
        d1 = dates_soil_start[idx]
        d2 = dates_soil_stop[idx]
        if np.min(soil.loc[d1:d2]) <= x:
            ids.append(idx)
    dates_soil_start_filtered = dates_soil_start[ids]
    dates_soil_stop_filtered = dates_soil_stop[ids]

    #df forsignificant rains.
    df_soil_output = pd.DataFrame.from_dict({"SoilStart": dates_soil_start_filtered, "SoilStop": dates_soil_stop_filtered})
    df_soil_output=df_soil_output.reset_index(drop='index')
    df_soil_output.reset_index(drop='index',inplace=True)
    print(f"We found {df_soil_output.shape[0]} Soiling Events with decay less than {x} ")

    if plot:
        print('The indication of the start of a Soil is presented with Bold line')
        print('The indication of the end of a Soil is presented with Uncontinious line')
        ax=df.soiling_derate.plot(figsize=(20,10),label='Soil Derate',color='green')
        for d in df_soil_output.SoilStart:
            ax.axvline(x=d, color='grey', linestyle='-')
        for d in df_soil_output.SoilStop:
            ax.axvline(x=d, color='grey', linestyle=':') 
        ax.set_title('Power Output', fontsize=8)
        plt.legend(fontsize=8)
        plt.show()
        
    return df_soil_output




def list_of_soil_index(df,df_soil_output,days):
    """
    Creates a list of discrete indexes corresponding to soiling events.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        df_soil_output (pandas.DataFrame): A DataFrame containing soiling event start and stop dates.
        days (int): The number of days to shift the index of soiling events.

    Returns:
        list: A list of lists, where each inner list contains indexes for a soiling event.
    """
    temp=df.reset_index()
    list_soil_index=[]
    for i in range(len(df_soil_output)):
        list_soil_index.append(list(range(temp[temp.timestamp==df_soil_output.SoilStart[i]].index[0]-days,
                                          temp[temp.timestamp==df_soil_output.SoilStop[i]].index[0])))
    lista_me_ta_index_apo_soil=[]
    for i in range(len(list_soil_index)):
        for j in range(len(list_soil_index[i])):
            lista_me_ta_index_apo_soil.append(list_soil_index[i][j])
    return lista_me_ta_index_apo_soil
    
def list_of_all_motifs_indexes(mi,new_population,row):
    """
    Creates a list of discrete indexes from found motifs.

    Args:
        mi: Motif indexes.
        new_population: Population of individuals.
        row: The index of each individual.

    Returns:
        list: A list of lists, where each inner list contains indexes for a motif.
    """
    lista_listwn=[]
    for mtyp in range(len(mi)):
        try1=[]
        for i in mi[mtyp]:
            try1.append(list(range(i,i+int(new_population[row,5]))))
        listamot=[]
        for i in range(len(try1)):
            for j in range(len(try1[i])):
                listamot.append(try1[i][j])

        lista_listwn.append(listamot)
    return lista_listwn

def list_of_soil_index_start(df,df_soil_output,days):
    """
    Creates a list of discrete indexes corresponding to soiling event start dates.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        df_soil_output (pandas.DataFrame): A DataFrame containing soiling event start and stop dates.
        days (int): The number of days to shift the index of soiling events.

    Returns:
        list: A list of indexes corresponding to soiling event start dates.
    """
    temp=df.reset_index()
    list_soil_index=[]
    for i in range(len(df_soil_output)):
        list_soil_index.append(temp[temp.timestamp==df_soil_output.SoilStart[i]].index[0]-days)
    return list_soil_index
     
def list_of_all_motifs_indexes_start(mi):
    """
    Creates a list of discrete indexes from found motifs.

    Args:
        mi: Motif indexes.

    Returns:
        list: A list of lists, where each inner list contains indexes for a motif.
    """
    lista_listwn=[]
    for mtyp in range(len(mi)):
        try1=[]
        for i in mi[mtyp]:
            try1.append(i)
       
        lista_listwn.append(try1)

    return lista_listwn

def reg_container(df,test_index,future_steps,pipeline,df_test,fit_features,pipeline_lower,target_features):
    """
    Fits regression models, stores predictions, and calculates MAPE values for different labels.

    Args:
        df: DataFrame with training data.
        test_index: Index for the test data.
        future_steps: Number of future steps to predict.
        pipeline: The primary regression model pipeline.
        df_test: DataFrame with test data.
        fit_features: Features used for fitting the model.
        pipeline_lower: A secondary regression model pipeline.
        target_features: Target features for prediction.

    Returns:
        p_list_pred: Predicted values for each label.
        p_list_pred_lower: Predicted values (secondary model) for each label.
        p_list: Ground truth values for each label.
        all_mapes: List of MAPE values for each label.
    """
    all_mapes = []
    p_list_pred = []
    p_list_pred_lower = []
    p_list = []
    for l in np.unique(df.label):
        try:
            unq_idx = test_index.drop_duplicates()
            temp = np.empty((unq_idx.shape[0], future_steps))
            temp[:] = 1e-6
            result_container = pd.DataFrame(temp.copy(), index=unq_idx)

            y_pred_test = pipeline.predict(df_test.loc[df_test.label==l][fit_features].values)
            result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
            result_container.loc[result_temp.index] = result_temp

            result_container_lower = pd.DataFrame(temp.copy(), index=unq_idx)
            y_pred_test_lower =pipeline_lower.predict(df_test.loc[df_test.label==l][fit_features].values)
            result_temp_lower = pd.DataFrame(y_pred_test_lower, index=df_test.loc[df_test.label==l].index)
            result_container_lower.loc[result_temp_lower.index] = result_temp_lower



            gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_test = df_test.loc[df_test.label==l][target_features]
            gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
            gt_container.loc[gt_temp.index] = gt_temp
            p_list_pred.append(result_container.copy())
            p_list_pred_lower.append(result_container_lower.copy())
            p_list.append(gt_container.copy())
            all_mapes.append(mape1(y_test, y_pred_test))
        except:
            print(l)
    return p_list_pred,p_list_pred_lower,p_list,all_mapes



def pred_gt_list(df,test_index,df_test,fit_features,target_features,pipeline):
    """
    Fits regression models, stores predictions, and calculates MAPE values for different labels.

    Args:
        df: DataFrame with training data.
        test_index: Index for the test data.
        df_test: DataFrame with test data.
        fit_features: Features used for fitting the model.
        target_features: Target features for prediction.
        pipeline: The primary regression model pipeline.

    Returns:
        p_list_pred: Predicted values for each label.
        p_list: Ground truth values for each label.
        all_mapes: List of MAPE values for each label.
    """
    p_list_pred=[]
    p_list=[]
    all_mapes=[]
    for l in np.unique(df.label):
        try:
            unq_idx = test_index.drop_duplicates()
            temp = np.empty((unq_idx.shape[0], future_steps))
            temp[:] = 1e-1
            result_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_pred_test =pipeline.predict(df_test.loc[df_test.label==l][fit_features].values)
            result_temp = pd.DataFrame(y_pred_test, index=df_test.loc[df_test.label==l].index)
            result_container.loc[result_temp.index] = result_temp
            gt_container = pd.DataFrame(temp.copy(), index=unq_idx)
            y_test = df_test.loc[df_test.label==l][target_features]
            gt_temp = pd.DataFrame(y_test, index=df_test.loc[df_test.label==l].index)
            gt_container.loc[gt_temp.index] = gt_temp
            p_list_pred.append(result_container.copy())
            p_list.append(gt_container.copy())
            all_mapes.append(mape1(y_test, y_pred_test))

        except:
            print(l)
    return p_list_pred,p_list,all_mapes
