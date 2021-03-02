#Define DateEncode as basic features such as day of week, month int, and year int.
#Define groupencode as placing into buckets
#Define mapencode as mapping values to a higher-level grouping

#Define relavent fields to model and their associated data types and feature engineering methods

colDict = {
    "COMP_CODE": {"Type":"Categorical", "Method":"1hot"}, #In
    #"REF_KEY1":{"Type":"Categorical", "Method":"1hot"},
    "AC_DOC_NR": {"Type":"ID", "Method":"N/A"},
    "PSTNG_DATE":{"Type":"Date", "Method":"DateEncode"},
    #"CREATEDON":{"Type":"Date", "Method":"DateEncode"},
    "DOC_DATE":{"Type":"Date", "Method":"DateEncode"},
    "PROFIT_CTR":{"Type":"Categorical", "Method":"1hot"},
    "AC_DOC_TYP":{"Type":"Categorical", "Method":"1hot"},
    "POST_KEY":{"Type":"Categorical", "Method":"1hot"},
    "V.VENDOR":{"Type":"Categorical", "Method":"MapEncode"},#Was creditor
    "V.NAME":{"Type":"Text", "Method":"Tokenize"}, #Was name
    "GL_ACCOUNT":{"Type":"Categorical", "Method":"1hot"},
    #"CUSTOMER":{"Type":"Categorical", "Method":"MapEncode"}, no longer included
    "POSTXT":{"Type":"Text", "Method":"Tokenize"},
    #"BIC_ZUSERNAM":{"Type":"Categorical", "Method":"MapEncode"}, no longer included
    "ZBKTXT":{"Type":"Text", "Method":"Tokenize"},
    "ZTCODE":{"TYpe":"Categorical", "Method":"1hot"},
    "DEB_CRE_USD":{"Type":"Numerical", "Method":"GroupEncode"},
    "V.BIC_ZTERMPAY":{"Type":"Categorical", "Method":"1hot"},
    "MATL_GROUP":{"Type":"Categorical", "Method":"1hot"},
    "ACCNT_GRPV":{"Type":"Categorical", "Method":"1hot"},
    "HCPClass":{"Type":"Categorical", "Method":"LabelEncode"}, #Target
    "Compliance Bucket":{"Type":"Categorical", "Method":"1hot"},
    "V.BI_ACCNT_TGRPV":{"Type":"Categorical", "Method":"1hot"} #Vendor type
}

oneHotFeatures = ["PROFIT_CTR","AC_DOC_TYP","POST_KEY","GL_ACCOUNT","ZTCODE", "MATL_GROUP", "ACCNT_GRPV","V.BI_ACCNT_TGRPV", "V.BIC_ZTERMPAY"]


#Set pandas datatypes for each field
dfValidateValidate['COMP_CODE'] = dfValidateValidate['COMP_CODE'].astype('object')
#dfValidateValidate['REF_KEY1'] = dfValidateValidate['REF_KEY1'].astype('object')
dfValidateValidate['AC_DOC_NR'] = dfValidateValidate['AC_DOC_NR'].astype('str')
dfValidateValidate['PSTNG_DATE'] = pd.to_datetime(dfValidateValidate['PSTNG_DATE'], format="%Y%m%d")
#dfValidateValidate['CREATEDON'] = pd.to_datetime(dfValidateValidate['CREATEDON'].astype('object'), format="%Y%m%d")
dfValidateValidate['DOC_DATE'] = pd.to_datetime(dfValidateValidate['DOC_DATE'], format="%Y%m%d")
dfValidateValidate['PROFIT_CTR'] = dfValidateValidate['PROFIT_CTR'].astype('str')
dfValidateValidate['AC_DOC_TYP'] = dfValidateValidate['AC_DOC_TYP'].astype('object')
dfValidateValidate['POST_KEY'] = dfValidateValidate['POST_KEY'].astype('str')
#dfValidateValidate_test['POST_KEY'] = dfValidateValidate_test['POST_KEY'].astype('int')
dfValidateValidate['V.VENDOR'] = dfValidateValidate['V.VENDOR'].astype('object') #was CREDITOR
dfValidate['V.NAME'] = dfValidate['V.NAME'].astype('object') #Was #name
dfValidate['GL_ACCOUNT'] = dfValidate['GL_ACCOUNT'].astype('int')
#dfValidate_test['GL_ACCOUNT'] = dfValidate_test['GL_ACCOUNT'].astype('object')
#dfValidate['CUSTOMER'] = dfValidate['CUSTOMER'].astype('object') no longer included
dfValidate['POSTXT'] = dfValidate['POSTXT'].astype('object')
#dfValidate['BIC_ZUSERNAM'] = dfValidate['BIC_ZUSERNAM'].astype('object') no longer included, but may be useful to bring back
dfValidate['ZBKTXT'] = dfValidate['ZBKTXT'].astype('object')
dfValidate['ZTCODE'] = dfValidate['ZTCODE'].astype('str')
dfValidate['DEB_CRE_USD'] = pd.to_numeric(dfValidate['DEB_CRE_USD'])
dfValidate['V.BIC_ZTERMPAY'] = dfValidate['V.BIC_ZTERMPAY'].astype('str') #no longer included, but may be useful to bring back
dfValidate['MATL_GROUP'] = dfValidate['MATL_GROUP'].astype('str')
dfValidate['ACCNT_GRPV'] = dfValidate['ACCNT_GRPV'].astype('str')
dfValidate['HCPClass'] = dfValidate['HCPClass'].astype('object')
#dfValidate['Compliance Bucket'] = dfValidate['Compliance Bucket'].astype("object")
dfValidate['V.BI_ACCNT_TGRPV'] = dfValidate['V.BI_ACCNT_TGRPV'].astype("str")




#Filter data for relavent rows
dfValidate = dfValidate[ ((dfValidate.POST_KEY == '40') | (dfValidate.POST_KEY == '81')) & (dfValidate.GL_ACCOUNT >= 40000000)]

dfValidate['GL_ACCOUNT'] = dfValidate['GL_ACCOUNT'].astype('str')
#Filter data for relavent rows
#dfValidate_test = dfValidate_test[ ((dfValidate_test.POST_KEY == 40) | (dfValidate_test.POST_KEY == 81)) & (dfValidate_test.GL_ACCOUNT >= 40000000)]

#Get list of relavent fields
colSelect = list(colDict.keys())
colSelect = ['COMP_CODE', 'AC_DOC_NR', 'PSTNG_DATE', 'DOC_DATE', 'PROFIT_CTR', 'AC_DOC_TYP', 'POST_KEY', 'V.VENDOR', 'V.NAME', 'GL_ACCOUNT', 'POSTXT', 'ZBKTXT' ,'ZTCODE', 'DEB_CRE_USD', 'V.BIC_ZTERMPAY', 'MATL_GROUP', 'ACCNT_GRPV', 'V.BI_ACCNT_TGRPV', 'HCPClass']


#dfValidate[colSelect]
dfValidate = dfValidate[colSelect]

#dfValidate_test[colSelect]


dfValidate