import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


pd.options.display.float_format = "{:,.2f}".format

st.set_page_config('GMV-Fees Reconciliation',':microscope:',layout='wide')

st.title(':microscope: GMV-Fees :blue[Reconciliation]')

st.text('Please upload the files to perform recon.')

st.subheader("File Upload and DataFrame Display")


def get_dataframe_name(file):
    """
    Generates a name for the DataFrame based on the file name.
    """
    file_name = file.name.split(".")[0]  # Get the file name without extension
    df_name = file_name.replace(" ", "_")  # Remove spaces and replace with underscores
    return df_name

def load_dataframe(file):
    """
    Loads the uploaded file into a Pandas DataFrame.
    """
    file_extension = file.name.split(".")[-1]

    if file_extension == "csv":
        df = pd.read_csv(file)
    elif file_extension == "xlsx":
        df = pd.read_excel(file)

    return df


st.cache()
def filter_dataframe(df: pd.DataFrame,key) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Add filters",value=True,key=key)

        if not modify:
            return df

        df = df.copy()

        key2 = key + '_'
        
        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Write {column} Name",key= f'{key}_text_widget'
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]         
    
        csv = df.to_csv().encode('utf-8')
           
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name= 'data.csv',
            mime='text/csv',
            )

        key2 = key + '_'
        if st.checkbox('Visualize Table',key=key2):
            st.dataframe(df,)

        
        return df



# Allow users to upload multiple files
uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
        
    # Create a dictionary to store dataframes
    dataframes = {}

    # Iterate through each uploaded file
    for file in uploaded_files:
        df_name = get_dataframe_name(file)
        df = load_dataframe(file)
        dataframes[df_name] = df


    # Reading Files into a Data frame
    nabis_invoices = dataframes['Aging_Nabis']
    aging_QBO_df = dataframes['AR_aging_report_QBO']
    aging_QBO_df = aging_QBO_df[['Date','Transaction Type','Num','Customer','Due Date','Amount','Open Balance','Memo/Description','Created By','Create Date','Last Modified By','Last Modified']]
    Report_zero_value = dataframes['Report_zero_value']
    revenue_df = dataframes['4500_Revenue_report_Consolidated']

    retool_df_2023 = dataframes['Retool_2023']
    retool_df_2022 = dataframes['Retool_2022']
    retool_df_2021 = dataframes['Retool_2021']
    retool_df_2020 = dataframes['Retool_2020']


    retool_df_2022 = retool_df_2022[['Order #','Delivery Date','RetLic','Brand DBA','Delivery Status','GMV','Order Credit','Order Discount','Order Surcharge','Excise Tax',
                                                'Payment Status','GMV Collected','Excise Tax Collected','Retailer','Distro Fee','Extra Fees','LineItem Discounts','Issue Reason','Factor Status']]

    retool_df_2020 = retool_df_2020[['Order #','Delivery Date','RetLic','Brand DBA','Delivery Status','GMV','Order Credit','Order Discount','Order Surcharge','Excise Tax',
                                                'Payment Status','GMV Collected','Excise Tax Collected','Retailer','Distro Fee','Extra Fees','LineItem Discounts','Issue Reason','Factor Status']]

    retool_df_2021 = retool_df_2021[['Order #','Delivery Date','RetLic','Brand DBA','Delivery Status','GMV','Order Credit','Order Discount','Order Surcharge','Excise Tax',
                                                'Payment Status','GMV Collected','Excise Tax Collected','Retailer','Distro Fee','Extra Fees','LineItem Discounts','Issue Reason','Factor Status']]

    retool_df_2023 = retool_df_2023[['Order #','Delivery Date','RetLic','Brand DBA','Delivery Status','GMV','Order Credit','Order Discount','Order Surcharge','Excise Tax',
                                                'Payment Status','GMV Collected','Excise Tax Collected','Retailer','Distro Fee','Extra Fees','LineItem Discounts','Issue Reason','Factor Status']]
                                                


    # Concatenate files into 1 
    retool_df = pd.concat([retool_df_2022,retool_df_2020,retool_df_2021,retool_df_2023],ignore_index=True)

    # Working with nabis aging file to standarize format and Data Types
    nabis_invoices['Delivery Date'] = pd.to_datetime(nabis_invoices['Delivery Date']) # Converts Date Data type Object to Datetime datatype
    nabis_invoices = nabis_invoices[['Overdue','Delivery Date','Order Number', 'Due', 'Subtotal',
        'Tax','Subtotal Collected', 'Tax Collected','Dispensary','Nabis Overdue Fee Status']] # Selecting only columns needed

    # Working with Aging QBO invoices file to standardize the format and Data Types 
    aging_QBO_df.dropna(subset=['Date'],inplace=True)
    aging_QBO_df['Num'] = aging_QBO_df['Num'].astype('str')
    aging_QBO_df = aging_QBO_df.loc[~aging_QBO_df['Num'].str.contains('-SH')].copy()
    aging_QBO_df['Date'] = pd.to_datetime(aging_QBO_df['Date'])
    aging_QBO_df['Num'] = aging_QBO_df['Num'].astype('str')
    aging_QBO_df['Invoice'] = aging_QBO_df['Num'].str.replace('([a-z][0-9]|[^\d])','',regex=True)
    aging_QBO_df = aging_QBO_df.loc[aging_QBO_df['Invoice']!=''].copy()
    aging_QBO_df['Invoice'] = aging_QBO_df['Invoice'].astype('int64')

    revenue_df['Transaction number'] = revenue_df['Transaction number'].astype('str')
    revenue_df['Invoice'] = revenue_df['Transaction number'].str.replace('([a-z][0-9]|[^\d])','',regex=True)
    revenue_df = revenue_df.loc[revenue_df['Invoice']!=''].copy()
    revenue_df['Invoice'] = revenue_df['Invoice'].astype('int64')
    revenue_df['Amount line'] = revenue_df['Amount line'].astype('str')
    revenue_df['Amount line'] = revenue_df['Amount line'].apply(lambda x: x.replace('$',''))
    revenue_df['Amount line'] = revenue_df['Amount line'].apply(lambda x: x.replace(',',''))
    revenue_df['Amount line'] = pd.to_numeric(revenue_df['Amount line'])

    Report_zero_value['Invoice number'] = Report_zero_value['Invoice number'].astype('str')
    Report_zero_value['Invoice'] = Report_zero_value['Invoice number'].str.replace('([a-z][0-9]|[^\d])','',regex=True)
    Report_zero_value = Report_zero_value.loc[Report_zero_value['Invoice']!=''].copy()
    Report_zero_value['Invoice'] = Report_zero_value['Invoice'].astype('int64')



    retool_df['Order_tot_amt'] = (retool_df['GMV'] + retool_df['Excise Tax'] - retool_df['Order Credit'] - retool_df['Order Discount'] - retool_df['Order Surcharge'] - retool_df['LineItem Discounts']).round(2)
    retool_df['Due_Amount'] = (retool_df['Order_tot_amt'] - retool_df['GMV Collected'] - retool_df['Excise Tax Collected']).round(2)


    # group Dataframes to get total invoice amount.
    group_aging_QBO_df = aging_QBO_df.groupby('Invoice')['Open Balance'].sum().reset_index()
    group_aging_QBO_df['Invoice'] = group_aging_QBO_df['Invoice'].astype('int')
    report_zero_value_gpd = Report_zero_value.groupby('Invoice')['Amount'].sum().reset_index()
    report_zero_value_gpd['Invoice'] = report_zero_value_gpd['Invoice'].astype('int')
    revenue_df_gpd = revenue_df.groupby('Invoice')['Amount line'].sum().reset_index()
    revenue_df_gpd['Invoice'] = revenue_df_gpd['Invoice'].astype('int')
  


    # Creating Dictionaries to cross match with Reetool
    nabis_invoices_dict = dict(zip(nabis_invoices['Order Number'],nabis_invoices['Due']))
    aging_QBO_dict = dict(zip(group_aging_QBO_df['Invoice'],group_aging_QBO_df['Open Balance']))
    Report_zero_value_dict = dict(zip(report_zero_value_gpd['Invoice'],report_zero_value_gpd['Amount']))
    revenue_df_dict = dict(zip(revenue_df_gpd['Invoice'],revenue_df_gpd['Amount line']))   

    

    retool_df['Nabis_Aging_Due'] = retool_df['Order #'].map(nabis_invoices_dict)
    retool_df['QBO_open_balance'] = retool_df['Order #'].map(aging_QBO_dict)
    retool_df['Aging_QBO_Amount'] = retool_df['Order #'].map(aging_QBO_dict)
    retool_df['zero_value_inv'] = retool_df['Order #'].map(Report_zero_value_dict)
    retool_df['Revenue Amt'] = retool_df['Order #'].map(revenue_df_dict)
    

    invoices_in_Qbo = retool_df['QBO_open_balance'].fillna('Not in QBO open Invs')
    retool_df['Nabis_Aging_Due'] = retool_df['Nabis_Aging_Due'].fillna('Not in Nabis Aging')
    retool_df['QBO_open_balance'] = retool_df['QBO_open_balance'].fillna(0.0)
    retool_df['Invoice in QBO'] = invoices_in_Qbo
    retool_df['Invoice_C_value'] = (retool_df['GMV'] - retool_df['Order Credit']	- retool_df['Order Discount'] -	retool_df['Order Surcharge'] - retool_df['Distro Fee'] - retool_df['Extra Fees'] - retool_df['LineItem Discounts']).round(2)
    retool_df['Invoice A'] = retool_df['Excise Tax']
    retool_df['Invoice B'] = (retool_df['Distro Fee'] + retool_df['Extra Fees']).round(2)
    retool_df['Invoice C'] = (retool_df['GMV'] - retool_df['Order Credit'] - retool_df['Order Discount'] - retool_df['Order Surcharge'] - retool_df['LineItem Discounts']).round(2)
    retool_df['Variance'] = (retool_df['Due_Amount'] - retool_df['QBO_open_balance']).round(2)

    retool_df_fltr = retool_df.loc[retool_df['Delivery Status'].str.contains('DELIVERED')].copy()


    aging_QBO_df['Invoice Breakdown'] = np.where(aging_QBO_df['Num'].str.contains('a'),'Invoice A',np.where(  
                                            (aging_QBO_df['Num'].str.contains('b')) | (aging_QBO_df['Num'].str.contains('b2')) , 'Invoce B', np.where(
                                            (aging_QBO_df['Num'].str.contains('c')) | (aging_QBO_df['Num'].str.contains('c1')) | (aging_QBO_df['Num'].str.contains('c2')), 'Invoice C', 'Single')))


    QBO_invoice_breakdown = aging_QBO_df.groupby(['Invoice','Invoice Breakdown']).agg({'Open Balance':'sum'}).reset_index()

    breakdown_invoce_df = QBO_invoice_breakdown.pivot(index='Invoice',columns='Invoice Breakdown',values='Open Balance').reset_index()
    breakdown_invoce_df.rename(columns={'Invoice':'QBO_Invoice', 'Invoce B':'QBO_Invoce B', 'Invoice A':'QBO_Invoice A', 'Invoice C':'QBO_Invoice C', 'Single':'QBO_Single'},inplace=True)
    breakdown_invoce_df.replace("NaN",np.nan,inplace=True)
    breakdown_invoce_df.fillna(0,inplace=True)

    retool_df_fltr = pd.merge(retool_df_fltr,breakdown_invoce_df,how='left',left_on='Order #',right_on='QBO_Invoice')

    retool_df_fltr['Comments_actions'] = np.where(retool_df_fltr['Variance'] == 0, "OK No Variance",
                                                np.where(retool_df_fltr['Variance'] == 0.01, "OK Small Ballance",
                                                        np.where(retool_df_fltr['Variance'] == -0.01, "OK Small Ballance",
                                                                    np.where(retool_df_fltr['Variance'] == retool_df_fltr['Invoice_C_value'], "OK Invoice C not generated yet",
                                                                             np.where((retool_df_fltr['Variance']+retool_df_fltr['QBO_Invoce B']) == 0, "OK, 90 days fees. Inv c2 and b open","Review")))))

    retool_df_fltr['Fees_revenue_QBO_comparison'] = np.where(retool_df_fltr['Revenue Amt'].round(2) == retool_df_fltr['Invoice B'].round(2), "OK No Variance","Review")
    
    retool_df_fltr['Order #'] = retool_df_fltr['Order #'].astype('str')


    filter_dataframe(retool_df_fltr,'gmv_recon')



