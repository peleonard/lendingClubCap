
# loading required Data Massaging and Machine Learning Package
library(data.table)
library(tidyverse)
library(caret)
library(zoo)
library(doParallel, quietly=TRUE, warn.conflicts=FALSE)

library(DMwR, quietly=TRUE, warn.conflicts=FALSE) # for smote implementation
library(purrr, quietly=TRUE, warn.conflicts=FALSE) # for functional programming (map)
library(pROC, quietly=TRUE, warn.conflicts=FALSE) # for AUC calculations
library(PRROC, quietly=TRUE, warn.conflicts=FALSE) # for Precision-Recall curve calculations

#### 1. Fast Parsing all Four Datasets ------------------------------------------

loanStat <- lapply(Sys.glob("data/LoanStats_2017Q*.csv"), 
                   fread, na.strings = "", skip = 1, fill=T)
names(loanStat) <- paste0("Q", 1:4)

rejStat <- lapply(Sys.glob("data/RejectStats_2017Q*.csv"), fread, na.strings = "", skip = 1)
names(rejStat) <- paste0("Q", 1:4)

loanStat$Q1
dim(loanStat$Q1)

rejStat$Q1
dim(rejStat$Q1)

## Check the overlapping information of the names of two datasets

# rejStat VS loanStat
# convert the name format to that of the loanStat (loan has been issued )
wordUnCap <- function(word) { paste0(tolower(substring(word, 1, 1)), 
                                     substring(word, 2)) }
# complicate word Uncap
cwUnCap <- function(cw, splitSep=' |_', concatSep = '_') {
  cw_split <- strsplit(cw, split=splitSep)
  cws <- unlist(lapply(cw_split, wordUnCap))
  paste0(cws, collapse = concatSep)
}
name_rej <- lapply(names(rejStat$Q1), cwUnCap)

## Match Conclusion
print(name_rej)
## amount_requested matches the loan amount
## application date no match
## loan_title matches the title
## risk_score matches the credit score columns information 
name_rej %in% names(loanStat$Q1) # zip code and policy code are in the big loanStat dataset 
grep('state', names(loanStat$Q1), value = T)
grep('emp', names(loanStat$Q1), value = T)





#### 2. Featurn Exploration and Cleansing -----------------------------------


## A careful check tells us they have the same variables 
lapply(loanStat, dim)
all.equal(names(loanStat[[1]]), names(loanStat[[2]]))
all.equal(names(loanStat[[2]]), names(loanStat[[3]]))
all.equal(names(loanStat[[3]]), names(loanStat[[4]]))
# They can be combined.

## Use the first quarter data as a test sample (lS: loan Sample dataset)
lS <- loanStat$Q1[, .SD]
# lS <- rbindlist(loanStat, use.names = TRUE)
N = nrow(lS)



### 0) (Last/Next) Payment Date and Loan Status: A Quasi-complete phenomenon---- 

##Missing Pattern of Last Payment Date
# a tiny proportion of loan's last pymnt is missing, all the loan is charged off, namely "bad"
table(lS$last_pymnt_d, useNA = "always")/N 
table(lS[is.na(last_pymnt_d), loan_status]) 
table(lS[is.na(next_pymnt_d), loan_status]) 

##Missing Pattern of Next Payment Date
# if next pymnt is missing, it is either charged off or fully paid for Q1 data
table(lS$next_pymnt_d, useNA = "always")/N
table(lS[!is.na(last_pymnt_d) & is.na(next_pymnt_d), loan_status])
table(lS[is.na(last_pymnt_d) & !is.na(next_pymnt_d), loan_status])


library(zoo)
lS[, issue_d_1 := as.Date(as.yearmon(issue_d, "%b-%Y"))]
lS[, ':='(issue_year = as.character(format(issue_d_1, "%Y")),
          issueMon = as.character(format(issue_d_1, "%m")))]
lS[, last_pymnt_d_1 := 
     as.Date(as.yearmon(last_pymnt_d, "%b-%Y"))][, lastPymntMon :=
                                                   as.character(format(last_pymnt_d_1, "%m"))]
lS[, next_pymnt_d_1 := 
     as.Date(as.yearmon(next_pymnt_d, "%b-%Y"))][, nextPymntMon :=
                                                   as.character(format(next_pymnt_d_1, "%m"))]


lS[, ':='(lastPymntFromIssue = last_pymnt_d_1 - issue_d_1, nextPymntFromIssue = next_pymnt_d_1 - issue_d_1)]

# generately speaking, the longer the last payment from issue, the less like ot 
qplot(x=loan_status, y=lastPymntFromIssue, data=lS, geom='boxplot')
qplot(x=loan_status, y=nextPymntFromIssue, data=lS, geom='boxplot')

## last pymnt from issue (in days) is a valid feature to predict whether the loan would be late.
# It is reasonable to discretize this feature, remember there is missing values
table(lS$lastPymntFromIssue, useNA = 'always')
table(lS$nextPymntFromIssue, useNA = 'always')


# 92: 3 months;   184: 6 months;   275: 9 months;   366: 1 year
# 457: 1 year and 3 months;   549: 1 year and 6 months;   639: 1 year and 9 months; 
# 730: 2 years;   <NA>: no pymnt
lS[, ':='(lastPymntFromIssueCat=
            as.character(cut(as.numeric(lastPymntFromIssue), c(-1, 0, 92, 184, 275, 366, 457, 549))),
          nextPymntFromIssueCat=
            as.character(cut(as.numeric(lastPymntFromIssue), c(-1, 0, 92, 184, 275, 366, 457, 549)))
)]
table(lS$lastPymntFromIssueCat, useNA = "always")
table(lS$nextPymntFromIssueCat, useNA = "always")

lS[, ':='(lastPymntFromIssueCat=
            ifelse(is.na(lastPymntFromIssueCat), 'no pymnt', lastPymntFromIssueCat),
          nextPymntFromIssueCat=
            ifelse(is.na(nextPymntFromIssueCat), 'no pymnt', nextPymntFromIssueCat)
)]

pymnt_date_VS_loan_status <- table(lS$lastPymntFromIssueCat, lS$loan_status)
# reorder the column and row
pymnt_date_VS_loan_status <- pymnt_date_VS_loan_status[c("(-1,0]", "(0,92]", "(92,184]", "(184,275]", "(275,366]",
                                                         "(366,457]", "no pymnt"),
                                                       c("Charged Off", "Default", "Late (31-120 days)",
                                                         "Late (16-30 days)", "In Grace Period", "Current",
                                                         "Fully Paid")]
round(100 * pymnt_date_VS_loan_status / apply(pymnt_date_VS_loan_status, 1, sum), 3)


qplot(x=loan_status, y=lastPymntFromIssueCat, data=lS, geom='count') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
qplot(x=loan_status, y=nextPymntFromIssueCat, data=lS, geom='count') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))





### 1) Explore and define the response: ----
##    We are going to predict the default rate, 
##    i.e. whether it will be the normal payment state: Current and Fully Paid, or not 
table(lS$loan_status, useNA = "always")/N

# 0: good loan, 1: late loan
lS[, loanStatus := as.factor(ifelse(loan_status %in% c("Current", "Fully Paid"), 0, 1))]
table(lS$loanStatus)
# The data is IMBALANCED, considering downsampling or upsampling


qplot(x=loanStatus, y=lastPymntFromIssueCat, data=lS, geom='count')
qplot(x=loanStatus, y=nextPymntFromIssueCat, data=lS, geom='count')

qplot(x=loanStatus, y=lastPymntFromIssue, data=lS, geom='boxplot')
qplot(x=loanStatus, y=nextPymntFromIssue, data=lS, geom='boxplot')

##The features chosen are:
# "lastPymntFromIssueCat" "nextPymntFromIssue"

modelFeatures <- c("loanStatus", "lastPymntFromIssueCat", "nextPymntFromIssueCat")
lS[, modelFeatures, with=F]




### 3) Feature Processing: Date columns ----

# ###"Cut" Function of discretize a continuous feature
# cut(1:4, c(0, 3, 5))
# typeof(cut(1:4, c(0, 3, 5)))
# typeof(as.character(cut(1:4, c(0, 3, 5))))

dateColsInd <- grep('_(d|date)$', names(lS))
(dateCols <- names(lS)[dateColsInd])
lS[, lapply(.SD, function(x) sum(is.na(x))), .SDcols=dateCols]


# check the date format
lS[, dateColsInd[1:3], with=F]
lS[, dateColsInd[4], with=F]
na.omit(lS[, dateColsInd[5:9], with=F])

# table(lS$last_credit_pull_d)

# parse date cols to Date Format
lS[, (paste0(names(lS)[dateColsInd][4:9], "_1"))
   :=lapply(.SD, function(x) as.Date(as.yearmon(x, format='%b-%Y'))), .SDcols=names(lS)[dateColsInd][4:9]]
table(lS$last_credit_pull_d_1, useNA = 'always')

summary(lS$last_credit_pull_d_1)
## Imputing NA with median
set(lS, i=which(is.na(lS$last_credit_pull_d_1)), 
    j=which(names(lS) == 'last_credit_pull_d_1'), value=median(lS$last_credit_pull_d_1, na.rm = T))

lS[, ':='(mthsLastCreditPullSinceIssue = as.integer((last_credit_pull_d_1 - issue_d_1)/30))]
# lS[, ':='(mthsSinceLastCreditPull = 
#             as.integer((max(last_credit_pull_d_1, na.rm = T) + 90 - last_credit_pull_d_1)/30))]
table(lS$mthsLastCreditPullSinceIssue, useNA = 'always')
table(lS$mthsSinceLastCreditPull, useNA = 'always')

# lS[, ':='(mthsLastCreditPullSinceIssue=
#             ifelse(is.na(mthsLastCreditPullSinceIssue), 'no avail',
#                    as.character(cut(mthsLastCreditPullSinceIssue, 
#                                     c(min(mthsLastCreditPullSinceIssue, na.rm = T) - 1,
#                                       quantile(mthsLastCreditPullSinceIssue, c(.1,.3,.7), na.rm = T),
#                                       max(mthsLastCreditPullSinceIssue, na.rm = T))))),
#           mthsSinceLastCreditPull=
#             ifelse(is.na(mthsSinceLastCreditPull), 'no avail',
#                    as.character(cut(mthsSinceLastCreditPull, 
#                                     c(min(mthsSinceLastCreditPull, na.rm = T) - 1,
#                                       quantile(mthsSinceLastCreditPull, c(.1, .3, .6), na.rm = T),
#                                       max(mthsSinceLastCreditPull, na.rm = T))))))]


# na.omit(lS[, .SD, .SDcols = grep("^hardship.*1$", names(lS))])
na.omit(lS[, .SD, .SDcols = grep("settlement.*1$", names(lS))])

## collapse ####
# colSelect <- function(col) { head(lS[, eval(as.name(col)]))}
# colSelect('hardship_start_date')

# lS[, mthsHardshipFromIssue
#    := ifelse(is.na(hardship_start_date_1), 'no_hardship', 
#              as.character(cut(as.integer((hardship_start_date_1-issue_d_1)/30),
#                               c(min(as.integer((hardship_start_date_1-issue_d_1)/30), na.rm = T) - 1,
#                                 quantile(as.integer((hardship_start_date_1-issue_d_1)/30), c(.1, .9), na.rm = T),
#                                 max(as.integer((hardship_start_date_1-issue_d_1)/30), na.rm = T)))))]
#####
dtTransDate <- function(col, startCol, newLevel) {
  newCol <- ifelse(is.na(col), newLevel, 
                   as.character(cut(as.integer((col-startCol)/30),
                                    c(min(as.integer((col-startCol)/30), na.rm = T) - 1,
                                      quantile(as.integer((col-startCol)/30), c(.1, .9), na.rm = T),
                                      max(as.integer((col-startCol)/30), na.rm = T)))))
  return(list(newCol))
}
lS[, mthsHardshipFromIssue:=dtTransDate(eval(as.name('hardship_start_date_1')),
                                        eval(as.name('issue_d_1')),
                                        'no_hardship')]
lS[, mthsSettlementFromIssue:=dtTransDate(eval(as.name('settlement_date_1')),
                                          eval(as.name('issue_d_1')),
                                          'no_settlement')]

# lS[is.na(hardship_end_date_1) & !is.na(hardship_start_date_1), ]
# lS[is.na(settlement_date_1) & !is.na(debt_settlement_flag_date_1), ]
lS[, ':='(mthsHardshipLength = as.integer((hardship_end_date_1 - hardship_start_date_1)/30))]
lS[, mthsHardshipLength:=ifelse(is.na(mthsHardshipLength), 0, mthsHardshipLength)]


yAndDatFeatures <- c(modelFeatures, 
                     "mthsLastCreditPullSinceIssue",
                     "mthsHardshipFromIssue", "mthsSettlementFromIssue",
                     "mthsHardshipLength")
str(lS[, yAndDatFeatures, with=F])

table(lS$mthsSettlementFromIssue)
table(lS$mthsHardshipLength)


### 4) Feature Processing: Categorical Features ----

# Since issue_d_1 (issue date in date format is the first column we generated)
newColsStartInd <- which("issue_d_1" == names(lS))

origCols <- names(lS)[1:(newColsStartInd-1)]
lS[, (origCols):=lapply(.SD, type.convert), .SDcols=origCols]

catCols <- origCols[!(unlist(lS[, lapply(.SD, is.numeric), .SDcols=1:(newColsStartInd-1)]))]
numCols <- origCols[unlist(lS[, lapply(.SD, is.numeric), .SDcols=1:(newColsStartInd-1)])]

## Remove Catetgorical Featurs with 1 level and all Date Columns in the previous seciton
catColsRemoved <- 
  catCols[unlist(lS[, lapply(.SD, function(x) length(unique(x))), .SDcols=catCols]) == 1]
# lS[, newColsStartInd:ncol(lS)]
lS[, c(catColsRemoved, dateCols):=NULL]

catCols <- setdiff(setdiff(catCols, catColsRemoved), dateCols)
lS[, catCols, with=F]

qplot(x=grade, y=loanStatus, data=lS, geom="count")

lS[,emp_length_clean
   :=unlist(strsplit(as.character(emp_length), " years*"))][,yrsEmpLengthCat
                                                            :=ifelse(emp_length_clean=="< 1", 0.5, 
                                                                     ifelse(emp_length_clean=="10+", 11, 
                                                                            ifelse(emp_length_clean=="n/a", 0,
                                                                                   as.numeric(emp_length))))]
lS[, yrsEmpLengthCat:=as.character(cut(yrsEmpLengthCat, 
                                       c(-1, 3, 6, 9, 12), 
                                       labels=c("<=3","(3,6]", "(6,9]", ">9")))]

qplot(x=loanStatus, y=title, data=lS, geom='count')
chisq.test(lS$loanStatus, lS$purpose)
table(lS$disbursement_method)


# default rate by addr_state
dt_drByState <- lS[, .(dr_by_state = sum(loanStatus==1)/.N), by=addr_state]
setorder(dt_drByState, dr_by_state)
dt_drByState[, addrStateCat:= cut(dr_by_state,
                                  c(min(dr_by_state)-.01, 
                                    quantile(dr_by_state, c(0.25, .5, .75, 1))),
                                  labels = c("low risk", "medium-low risk", 
                                             "medium-high risk", "high risk"))]
dt_drByState[, dr_by_state:=NULL]
lS <- merge(lS, dt_drByState, by='addr_state', all.x = T)

# default rate by zip_code
dt_drByZc <- lS[, .(dr_by_zc = sum(loanStatus==1)/.N), by=zip_code]
setorder(dt_drByZc, dr_by_zc)
dt_drByZc[, zipCodeCat:= cut(dr_by_zc,
                             c(min(dr_by_zc)-.01, 
                               quantile(dr_by_zc, c(.2, .4, .6, .8, 1))),
                             labels = c("low risk", "medium-low risk", "medium risk",
                                        "medium-high risk", "high risk"))]
dt_drByZc[, dr_by_zc:=NULL]
lS <- merge(lS, dt_drByZc, by='zip_code', all.x = T)


# qplot(x=verification_status, y=intRate, data=lS, geom='boxplot')  # intRate is defined in section 2 3)
table(lS$verification_status, y=lS$loanStatus)

table(lS$verification_status)
## deal with ".*_joint" features
sum(is.na(lS$verification_status))
lS[!is.na(verification_status_joint), c('verification_status', 'verification_status_joint')]

lS[, verificationStatus:=ifelse(!is.na(verification_status_joint), 
                                verification_status_joint, verification_status)]

table(lS$mthsHardshipFromIssue, lS$hardship_flag)
table(lS$hardship_status)
table(lS$disbursement_method)
table(lS$debt_settlement_flag, lS$settlement_status)

lS[, ':='(hardshipFlag = ifelse(is.na(hardship_flag), 'no_hs_flag', hardship_flag),
          disbursementMethod = ifelse(is.na(disbursement_method), 'no_db_method', 
                                      disbursement_method),
          settlementStatus = ifelse(is.na(settlement_status), 'no_st_status', settlement_status))]

catFeatures <- c('grade', 'term', 'yrsEmpLengthCat', 'home_ownership', 
                 'addrStateCat', 'zipCodeCat', 'verificationStatus',
                 'pymnt_plan', 'purpose', 'title', 'initial_list_status', 'application_type',
                 'hardshipFlag','disbursementMethod', 'debt_settlement_flag', 'settlementStatus')
str(lS[, catFeatures, with=F])
lS[, c(yAndDatFeatures, catFeatures), with=F]




### 5) Feature Processing: Numerical Features ----


# Numerical Featurs obtained after cleaning of the inferred discrete Features
lS[, ':='(intRate = as.numeric(unlist(strsplit(as.character(int_rate), "%"))), 
          revolUtil = ifelse(!is.na(sec_app_revol_util), sec_app_revol_util/100,
                             as.numeric(unlist(strsplit(as.character(revol_util), "%")))) )]
# lS[, lapply(.SD, summary), .SDcols = c("intRate", "revolUtil")]

lS[, mthsFromEarliestCrLineToIssue
   :=ifelse(!is.na(sec_app_earliest_cr_line), 
            as.integer((issue_d_1 - (as.Date(as.yearmon(sec_app_earliest_cr_line, format="%b-%Y"))))/30),
            as.integer((issue_d_1 - (as.Date(as.yearmon(earliest_cr_line, format="%b-%Y"))))/30)) ]


numCols
grep('_joint', names(lS), value=T)
lS[, ':='(annualInc = ifelse(!is.na(annual_inc_joint), annual_inc_joint, annual_inc),
          dti = ifelse(!is.na(dti_joint), dti_joint, dti),
          revolBal = ifelse(!is.na(revol_bal_joint), revol_bal_joint, revol_bal))]


# features Handled
numFsHandled_0 <- c('int_rate', 'revol_util', 'earliest_cr_line', 
                    c('annual_inc', 'dti', 'revol_bal'), 
                    paste0(c('annual_inc', 'dti', 'revol_bal'), '_joint'))


lS$mthsFromEarliestCrLineToIssue %>% summary()

## Potential Outliers for Features provided by  ----

##annual income
lS$annual_inc %>% summary() 
lS$annual_inc %>% boxplot()
# There are more outliers in "default" status, it is reasonable to beliveve the extremely high income 
# is fake or system error, it is not "missing" at random
qplot(x=loanStatus, y=annual_inc, data=lS, geom='boxplot')

log(lS$annual_inc + 1) %>% summary() %>% hist()

##dti
lS$dti %>% summary() 
lS$dti %>% boxplot()  
qplot(x=loanStatus, y=dti, data=lS, geom='boxplot')

##revol_balance
lS$revol_bal %>% summary() 
lS$revol_bal %>% boxplot()  
qplot(x=loanStatus, y=revol_bal, data=lS, geom='boxplot')


lS[, ':='(annualInclog10 = log10(annualInc + 1),
          revolBallog10 = log10(revolBal + 1))]

catFeatures_0 <- c("mthsFromEarliestCrLineToIssue")
numFeatures_0 <- c("annualInclog10", "dti", "revolBallog10")



# Missing values in Numerical Features
lS[, lapply(.SD, function(x) sum(is.na(x))/nrow(lS)), .SDcols = numCols]

# Missing Values belong to mainly to the following THREE patterns, handling one-by-one

## a) 'mths_since_.*' features ----

mthsSinceFeatures <- grep('^mths_since_', names(lS), value=T)
lS[, lapply(.SD, summary), .SDcols = mthsSinceFeatures]
sum(lS[is.na(mths_since_last_delinq), delinq_2yrs]==0)
lS[, mthsSinceLastDelinq:=ifelse(is.na(mths_since_last_delinq), 1000, mths_since_last_delinq)]


###N.B:  'mthsSinceLastMajorDerog', is handled in the section b) 'sec_app' features
lS[, c('mthsSinceLastDelinq', 'mthsSinceLastRecord',
       'mthsSinceRcntIl', 'mthsSinceRecentBc', 'mthsSinceRecentBcDlq', 
       'mthsSinceRecentInq', 'mthsSinceRecentRevolDelinq') 
   := lapply(.SD, function(col) ifelse(is.na(col), 'not_avail_mths', 
                                       as.character(cut(col, c(min(col, na.rm = T)-1,
                                                               quantile(col, c(.1,.9), na.rm = T),
                                                               max(col, na.rm = T)))))),
   .SDcols = setdiff(mthsSinceFeatures, 'mths_since_last_major_derog')]

moSinFeatures <- grep('^mo_sin_', numCols, value=T)
lS[, lapply(.SD, function(x) sum(is.na(x))/nrow(lS)), .SDcols=moSinFeatures]
summary(lS$mo_sin_old_il_acct)
summary(lS$mo_sin_old_rev_tl_op)
lS[, c('moSinOldIlAcct', 'moSinOldRevTlOp', 'moSinRcntRevTlOp', 'moSinRcntTl')
   := lapply(.SD, function(col) ifelse(is.na(col), 'not_avail_mo', 
                                       as.character(cut(col, c(min(col, na.rm = T)-1,
                                                               quantile(col, c(.1,.5,.9), na.rm = T),
                                                               max(col, na.rm = T)))))),
   .SDcols=moSinFeatures]

numFsHandled_a <- c(mthsSinceFeatures, moSinFeatures)
catFeatures_a <- c(c('mthsSinceLastDelinq', 'mthsSinceLastRecord', 
                     'mthsSinceRcntIl', 'mthsSinceRecentBc', 'mthsSinceRecentBcDlq', 
                     'mthsSinceRecentInq', 'mthsSinceRecentRevolDelinq'),
                   c('moSinOldIlAcct', 'moSinOldRevTlOp', 'moSinRcntRevTlOp', 'moSinRcntTl'))
lS[, catFeatures_a, with=F]



## b) 'sec_app_.*' features ----

#Special 'sec_app_' features
lS[, mthsSinceLastMajorDerog:=ifelse(!is.na(sec_app_mths_since_last_major_derog), 
                                     sec_app_mths_since_last_major_derog, mths_since_last_major_derog)]
lS[, mthsSinceLastMajorDerog:=ifelse(is.na(mthsSinceLastMajorDerog), 'no_avail_mths',
                                     as.character(cut(mthsSinceLastMajorDerog,
                                                      c(min(mthsSinceLastMajorDerog,na.rm=T)-1,
                                                        quantile(mthsSinceLastMajorDerog, c(.1,.5,.9), na.rm=T),
                                                        max(mthsSinceLastMajorDerog,na.rm=T)))))]
table(lS$mthsSinceLastMajorDerog)
# summary(lS$sec_app_mths_since_last_major_derog)

secappFeatures <- grep('^sec_app_', names(lS), value=T)

sec_app_Cols <- setdiff(secappFeatures, 
                        c('sec_app_earliest_cr_line', 'sec_app_revol_util', 
                          'sec_app_mths_since_last_major_derog'))

lS[, lapply(.SD, function(x) sum(is.na(x))/nrow(lS)), .SDcols = sec_app_Cols]
lS[, lapply(.SD, function(x) sum(is.na(x))/nrow(lS)), 
   .SDcols = sapply(strsplit(sec_app_Cols, '^sec_app_'), '[', 2)]

secApp_fun <- function(col, secCol) {
  newCol <- ifelse(!is.na(secCol), secCol, col)
  return(list(newCol))
}

lS[, inqLast6mths:=secApp_fun(eval(as.name('inq_last_6mths')), eval(as.name('sec_app_inq_last_6mths')))]
lS[, mortAcc:=secApp_fun(eval(as.name('mort_acc')), eval(as.name('sec_app_mort_acc')))]
lS[, openAcc:=secApp_fun(eval(as.name('open_acc')), eval(as.name('sec_app_open_acc')))]
lS[, openActIl:=secApp_fun(eval(as.name('open_act_il')), eval(as.name('sec_app_open_act_il')))]
lS[, numRevAccts:=secApp_fun(eval(as.name('num_rev_accts')), eval(as.name('sec_app_num_rev_accts')))]
lS[, chargeoffWithin12Mths:=secApp_fun(eval(as.name('chargeoff_within_12_mths')), 
                                       eval(as.name('sec_app_chargeoff_within_12_mths')))]
lS[, collections12MthsExMed:=secApp_fun(eval(as.name('collections_12_mths_ex_med')), 
                                        eval(as.name('sec_app_collections_12_mths_ex_med')))]


lS[, lapply(.SD, summary), .SDcols = sec_app_Cols]
summary(lS$sec_app_earliest_cr_line)
summary(lS$sec_app_inq_last_6mths)

head(lS$revol_util)
summary(lS$revolUtil)
summary(lS$sec_app_revol_util)


numFsHandled_b <- c(secappFeatures, sapply(strsplit(secappFeatures, '^sec_app_'), '[', 2))
catFeatures_b <- c('mthsSinceLastMajorDerog')
numFeatures_b <- c('inqLast6mths', 'mortAcc', 'openAcc', 
                   'openActIl', 'numRevAccts', 'chargeoffWithin12Mths', 'collections12MthsExMed')
lS[, lapply(.SD, summary), .SDcols=numFeatures_b]



## c) 'hardship' and 'settlement' features ----
summary(lS$deferral_term) # convert to Categorical
lS[ ,deferralTermCat := as.factor(ifelse(is.na(deferral_term), 0, 1))]

catFeatures_c <- c('deferralTermCat') 

hsFeatures <- grep('^hardship_', numCols, value=T)
smFeatures <- grep('^settlement_', numCols, value=T)

lS[, lapply(.SD, summary), .SDcols = hsFeatures]
lS[, c("hsAmount", "hsLength", "hsDpd", "hsPayoffBalanceAmount", "hsLastPaymentAmount")
   :=lapply(.SD, function(col) ifelse(is.na(col), 0, col)), 
   .SDcols = hsFeatures]

lS[, lapply(.SD, summary), .SDcols = smFeatures]
lS[, c("smAmount", "smPercentage", "smTerm")
   :=lapply(.SD, function(col) ifelse(is.na(col), 0, col)), 
   .SDcols = smFeatures]

numFsHandled_c <- c('deferral_term', 
                    hsFeatures,
                    smFeatures)
numFeatures_c <- c("hsAmount", "hsLength", "hsDpd", "hsPayoffBalanceAmount", "hsLastPaymentAmount",
                   "smAmount", "smPercentage", "smTerm")




## d) other features ----

lS[, lapply(.SD, function(x) sum(is.na(x))/nrow(lS)), 
   .SDcols = setdiff(numCols, c(numFsHandled_0, numFsHandled_a, numFsHandled_b, numFsHandled_c))]
summary(lS[is.na(il_util), c('total_bal_il', 'total_il_high_credit_limit'), with=F])

lS[, ilUtil:=ifelse(is.na(il_util)&!is.na(total_il_high_credit_limit), 
                    total_bal_il/total_il_high_credit_limit, il_util)]
lS[, ilUtilCat:=ifelse(is.na(ilUtil), 'no installment', 
                       as.character(cut(ilUtil, c(min(ilUtil, na.rm=T) - 1,
                                                  quantile(ilUtil, c(.1, .5, .9), na.rm=T),
                                                  max(ilUtil, na.rm=T)))))]
table(lS$ilUtilCat)

summary(lS$all_util)
lS[, allUtil:=ifelse(is.na(all_util), median(all_util, na.rm = T), all_util)]
summary(lS$allUtil)


summary(lS$bc_open_to_buy)
summary(lS$bc_util)
lS[, ':='(bcOpenToBuyCat=ifelse(is.na(bc_open_to_buy), 'no bc open-to-buy', 
                                as.character(cut(bc_open_to_buy, c(min(bc_open_to_buy, na.rm=T) - 1,
                                                                   quantile(bc_open_to_buy, c(.1, .5, .9), na.rm=T),
                                                                   max(bc_open_to_buy, na.rm=T))))),
          bcUtilCat=ifelse(is.na(bc_util), 'no bc_util', 
                           as.character(cut(bc_util, c(min(bc_util, na.rm=T) - 1,
                                                       quantile(bc_util, c(.1, .5, .9), na.rm=T),
                                                       max(bc_util, na.rm=T)))))
)]

table(lS$num_tl_120dpd_2m)
lS[, numTl120dpd2mCat:=ifelse(is.na(num_tl_120dpd_2m), 'no avail', as.factor(num_tl_120dpd_2m))]
table(lS$numTl120dpd2mCat)

table(lS$percent_bc_gt_75, useNA = 'always')
lS[, percentBcGt75Cat:=ifelse(is.na(percent_bc_gt_75), 'no avail', 
                              as.character(cut(percent_bc_gt_75, c(min(percent_bc_gt_75, na.rm=T) - 1,
                                                                   quantile(percent_bc_gt_75, c(.1, .5, .8), na.rm=T),
                                                                   max(percent_bc_gt_75, na.rm=T)))))]

table(lS$orig_projected_additional_accrued_interest)

lS[, origProjectedAdditionalAccruedInterest
   :=ifelse(is.na(orig_projected_additional_accrued_interest), 
            0, orig_projected_additional_accrued_interest)]


numFsHandled_d <- c('il_util', 'all_util', 'bc_open_to_buy', 'bc_util', 'num_tl_120dpd_2m', 'percent_bc_gt_75',
                    'orig_projected_additional_accrued_interest')

catFeatures_d <- c('ilUtilCat', 'bcOpenToBuyCat', 'bcUtilCat', 'numTl120dpd2mCat', 'percentBcGt75Cat')
numFeatures_d <- c('allUtil', 'origProjectedAdditionalAccruedInterest')


## e) numerical features remained ----

## Remove Catetgorical Featurs with 1 level and all Date Columns in the previous seciton
numColsRemoved <- 
  numCols[unlist(lS[, lapply(.SD, function(x) length(unique(x))), .SDcols=numCols]) <= 1]

numCols_remained <- setdiff(numCols, c(numFsHandled_0, 
                                       numFsHandled_a, numFsHandled_b, 
                                       numFsHandled_c, numFsHandled_d,
                                       numColsRemoved))

lS[, lapply(.SD, function(x) sum(is.na(x))/nrow(lS)), .SDcols = numCols_remained]


lS_num_melt <- melt(lS[,c("loanStatus", numCols_remained),with=F], id.vars='loanStatus')
p <- ggplot(data = lS_num_melt, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=loanStatus))
p + facet_wrap( ~ variable, scales="free")

hist(lS$out_prncp)
hist(log(lS$out_prncp+1))
boxplot(log(lS$out_prncp+1))


(numCols_log10Trans <- numCols_remained[unlist(lS[, lapply(.SD, max), 
                                                  .SDcols=numCols_remained])>1500])


lS[, c("loanAmntlog10", "fundedAmntlog10", "fundedAmntInvlog10",
       "installmentlog10", "outPrncplog10", "outPrncpInvlog10",
       "totalPymntlog10", "totalPymntInvlog10", "totRecPrncplog10",
       "totalRecIntlog10", "recoverieslog10", "collectionRecoveryFeelog10",
       "lastPymntAmntlog10", "totCollAmntlog10", "TotCurBallog10", 
       "TotalBalIllog10", "maxBalBc", "totalRevHiLimlog10",
       "avgCurBallog10", "delinqAmntlog10", "totHiCredLimlog10",
       "totalBalExMortlog10", "totalBcLimitlog10", 
       "totalIlHighCreditLimitlog10") := lapply(.SD, function(x) log10(x+1)), 
   .SDcols = numCols_log10Trans]

numFeatures_e = c("loanAmntlog10", "fundedAmntlog10", "fundedAmntInvlog10",
                  "installmentlog10", "outPrncplog10", "outPrncpInvlog10",
                  "totalPymntlog10", "totalPymntInvlog10", "totRecPrncplog10",
                  "totalRecIntlog10", "recoverieslog10", "collectionRecoveryFeelog10",
                  "lastPymntAmntlog10", "totCollAmntlog10", "TotCurBallog10", 
                  "TotalBalIllog10", "maxBalBc", "totalRevHiLimlog10",
                  "avgCurBallog10", "delinqAmntlog10", "totHiCredLimlog10",
                  "totalBalExMortlog10", "totalBcLimitlog10", 
                  "totalIlHighCreditLimitlog10")

numFeatures_z = setdiff(numCols_remained, numCols_log10Trans)



hist(log(lS$acc_open_past_24mths+1))


## f) Final Combination and Write CSV to disk ----

catFeatures_final <- c(catFeatures, catFeatures_0, catFeatures_a, catFeatures_b, catFeatures_c, catFeatures_d)
numFeatures_final <- c(numFeatures_0, numFeatures_b, numFeatures_c, numFeatures_d, numFeatures_e, numFeatures_z)
lS[, lapply(.SD, function(x) length(unique(x))), .SDcols=numFeatures_final]

model_Cols <- c(yAndDatFeatures, catFeatures_final, numFeatures_final)

lS_model <- lS[, model_Cols, with=F]
lS_model[, lapply(.SD, function(x) sum(is.na(x))/nrow(lS_model))]

str(lS_model[, 1:56])
str(lS_model[, 57:112])


#numerical and discrete features simple sifting

lS_m1 <- lS_model[, .SD]
for (col in numFeatures_final) {
  p.val <- t.test(as.formula(paste0(col, " ~ loanStatus")), data = lS_m1)$p.value
  if (p.val >= .05) { lS_m1[, (col):=NULL]}
}

for(col in catFeatures_final) {
  p.val <- chisq.test(x = lS_m1[, col, with=F], y = lS_m1$loanStatus)$p.value
  if(p.val >= 0.05) { lS_m1[, (col):=NULL] }
}

## g) Convert all character columns to Fators----
lS_m1_cats <- names(lS_m1)[unlist(lS_m1[, lapply(.SD, is.character)])]
lS_m1_levels <- names(lS_m1)[unlist(lS_m1[, lapply(.SD, function(x) length(unique(x)))]) <= 3]

lS_m1[, unique(c(lS_m1_cats, lS_m1_levels))
      :=lapply(.SD, as.factor), .SDcols=unique(c(lS_m1_cats, lS_m1_levels))]

str(lS_m1)
dim(lS_m1)

fwrite(lS_m1, "loanStat2017Q1_model_2nd.csv")



#### 3. Model ----------------------------------------------------------------


lS_m1 <- fread("loanStat2017Q1_model_2nd.csv")

lS_m1_col1 <- unique(c(names(lS_m1)[unlist(lS_m1[, lapply(.SD, is.character)])],
                       names(lS_m1)[unlist(lS_m1[, lapply(.SD, function(x) length(unique(x)))])<=3]
))
lS_m1[, (lS_m1_col1):=lapply(.SD, as.factor), .SDcols=lS_m1_col1]

table(lS_m1$lastPymntFromIssueCat)
table(lS_m1$mthsHardshipLength)
str(lS_m1)


### 1) Global EDA ----
library(GGally)
ggpairs(lS_m1[, c(1, 2), with=F], aes(colour = loanStatus, alpha = 0.4))
ggpairs(lS_m1[, 1:5], aes(colour = loanStatus, alpha = 0.4))
ggpairs(lS_m1[, c(1, 89, 90), with=F], aes(colour = loanStatus, alpha = 0.4))


library(corrplot)
lS_m1_numCols <- names(lS_m1)[unlist(lS_m1[, lapply(.SD, is.numeric)])]
M1 = cor(as.matrix(lS_m1[, lS_m1_numCols, with=F]))
corrplot(M1, method = "circle", tl.cex=.5, tl.srt=70)


### 2) Modeling ----

## a) stratified train test split ----

# rectify the name of response to 'Class"
names(lS_m1)[names(lS_m1)=='loanStatus'] <- 'Class'
lS_m1[, Class:=as.factor(ifelse(Class==0, "Good", "Bad"))]

table(lS_m1$Class)
str(lS_m1$Class)
summary(as.numeric(lS_m1$Class))


set.seed(998)
inTraining <- createDataPartition(lS_m1$Class, p = .75, list = FALSE)
training <- lS_m1[ inTraining,]
testing  <- lS_m1[-inTraining,]

dim(training)
str(training$Class)

## b) different models ---- 

cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)

##b1)gbm ---- 
library(gbm)
gbm_grid <- expand.grid(n.trees = seq(100, 150, by=50), interaction.depth = 2,
                        shrinkage = .1,n.minobsinnode=seq(9,11,by=2)) 

# set seed to run fully reproducible model in parallel mode using caret          
set.seed(825)
gbm_seeds <- vector(mode = "list", length = 5) # length is = (n_repeats*nresampling)+1
for(i in 1:4) gbm_seeds[[i]] <- sample.int(n=1000, 4) # ...the number of tuning parameter...
gbm_seeds[[5]]<-sample.int(1000, 1) # for the last model

## 'number'-fold CV, repeat 'repeats' times
gbm_fitControl <- trainControl(method = "repeatedcv", 
                               number = 2, repeats = 2,
                               seeds = gbm_seeds,
                               ## Estimate class probabilities
                               classProbs = TRUE,
                               ## Evaluate performance using 
                               ## the following function
                               summaryFunction = twoClassSummary)

gbmFit1 <- train(Class ~ ., data = training, 
                 method = "gbm", 
                 trControl = gbm_fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 tuneGrid = gbm_grid, verbose = FALSE,
                 metric = "ROC")
# plot(gbmFit1, metric = 'Kappa')
gbmFit1


gbm_pred <- predict(gbmFit1, testing, type='raw')
# several manually calculated statitics
# testing sensitivity 
(gbm_recall <- sum(gbm_pred=="Bad" & gbm_pred==testing$Class)/sum(testing$Class=="Bad"))
# testing positive predicted value
(gbm_precision <- sum(gbm_pred=="Bad" & gbm_pred==testing$Class)/sum(gbm_pred=="Bad") )
# testing specificity
(gbm_specificity <- sum(gbm_pred=="Good" & gbm_pred==testing$Class)/sum(testing$Class=="Good"))

confusionMatrix(table(gbm_pred, testing$Class))


## precision-recall curve: auprc(area under precision and recall curve)
# model = gbmFit1
# data = testing
calc_auprc <- function(model, data){
  
  index_bad <- data$Class == "Bad"
  index_good <- data$Class == "Good"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$Bad[index_bad], predictions$Bad[index_good], curve = TRUE)
  
}
calc_auprc(gbmFit1, testing)
plot(calc_auprc(gbmFit1, testing))

## Feature Importance
gbmImp <- varImp(gbmFit1, scale = FALSE)
gbmImp
plot(gbmImp, top=20, cex=.8)

# roc_imp <- filterVarImp(x = training[, -1], y = training$Class)
# head(roc_imp)


##Random Forests ----

####  THE ORIGINAL 'rf' do not have enough parameters to tune:
####  One can either use 'ranger' instead, or seek to 
## https://stackoverflow.com/questions/38625493/tuning-two-parameters-for-random-forest-in-caret-package


# library(randomForest, quietly=TRUE, warn.conflicts=FALSE)
# rf_grid <- expand.grid(mtry=seq(7, 10, by = 3),
#                        nodesize=10 ) 
# 
# # set seed to run fully reproducible model in parallel mode using caret          
# set.seed(826)
# rf_seeds <- vector(mode = "list", length = 5) # length is = (n_repeats*nresampling)+1
# for(i in 1:4) rf_seeds[[i]] <- sample.int(n=1000, 2) # ...the number of tuning parameter...
# rf_seeds[[5]]<-sample.int(1000, 1) # for the last model
# 
# ## 'number'-fold CV, repeat 'repeats' times
# rf_fitControl <- trainControl(method = "repeatedcv", 
#                               number = 2, repeats = 2,
#                               seeds = rf_seeds,
#                               ## Estimate class probabilities
#                               classProbs = TRUE,
#                               ## Evaluate performance using 
#                               ## the following function
#                               summaryFunction = twoClassSummary)
# 
# rfFit1 <- train(Class ~ ., data = training, 
#                  method = "rf", 
#                  trControl = rf_fitControl,
#                  ## This last option is actually one
#                  ## for randomForest() that passes through
#                  tuneGrid = rf_grid, verbose = FALSE,
#                  metric = "ROC")
# # plot(gbmFit1, metric = 'Kappa')
# rfFit1
# 
# 
# rf_pred <- predict(rfFit1, testing, type='raw')
# # several manually calculated statitics
# # testing sensitivity 
# (rf_recall <- sum(rf_pred=="Bad" & rf_pred==testing$Class)/sum(testing$Class=="Bad"))
# # testing positive predicted value
# (rf_precision <- sum(rf_pred=="Bad" & rf_pred==testing$Class)/sum(rf_pred=="Bad") )
# # testing specificity
# (rf_specificity <- sum(rf_pred=="Good" & rf_pred==testing$Class)/sum(testing$Class=="Good"))
# 
# confusionMatrix(table(rf_pred, testing$Class))
# 
# 
# ## precision-recall curve: auprc(area under precision and recall curve)
# # model = gbmFit1
# # data = testing
# calc_auprc <- function(model, data){
#   
#   index_bad <- data$Class == "Bad"
#   index_good <- data$Class == "Good"
#   
#   predictions <- predict(model, data, type = "prob")
#   
#   pr.curve(predictions$Bad[index_bad], predictions$Bad[index_good], curve = TRUE)
#   
# }
# calc_auprc(rfFit1, testing)
# plot(calc_auprc(rfFit1, testing))
# 
# ## Feature Importance
# rfImp <- varImp(rfFit1, scale = FALSE)
# rfImp
# plot(rfImp, top=20, cex=.8)





##b2)ranger ----
library(ranger, quietly=TRUE, warn.conflicts=FALSE)
library(e1071)
ranger_grid <- expand.grid(mtry=10,
                           splitrule = 'gini',
                           min.node.size = 20
) 


# set seed to run fully reproducible model in parallel mode using caret          
set.seed(827)
ranger_seeds <- vector(mode = "list", length = 4) # length is = (n_repeats*nresampling)+1
for(i in 1:4) ranger_seeds[[i]] <- sample.int(n=1000, 1) # ...the number of tuning parameter...
ranger_seeds[[5]]<-sample.int(1000, 1) # for the last model

## 'number'-fold CV, repeat 'repeats' times
ranger_fitControl <- trainControl(method = "repeatedcv", 
                                  number = 2, repeats = 2,
                                  seeds = ranger_seeds,
                                  ## Estimate class probabilities
                                  classProbs = TRUE,
                                  ## Evaluate performance using 
                                  ## the following function
                                  summaryFunction = twoClassSummary)

rangerFit1 <- train(Class ~ ., data = training, 
                    method = "ranger", 
                    trControl = ranger_fitControl,
                    importance = 'impurity',
                    ## This last option is actually one
                    ## for ranger() that passes through
                    tuneGrid = ranger_grid, verbose = FALSE,
                    metric = "ROC")
# plot(gbmFit1, metric = 'Kappa')
rangerFit1


ranger_pred <- predict(rangerFit1, testing, type='raw')
# several manually calculated statitics
# testing sensitivity 
(ranger_recall <- sum(ranger_pred=="Bad" & ranger_pred==testing$Class)/sum(testing$Class=="Bad"))
# testing positive predicted value
(ranger_precision <- sum(ranger_pred=="Bad" & ranger_pred==testing$Class)/sum(ranger_pred=="Bad") )
# testing specificity
(ranger_specificity <- sum(ranger_pred=="Good" & ranger_pred==testing$Class)/sum(testing$Class=="Good"))

confusionMatrix(table(ranger_pred, testing$Class))


## precision-recall curve: auprc(area under precision and recall curve)
# model = rangerFit1
# data = testing
calc_auprc <- function(model, data){
  
  index_bad <- data$Class == "Bad"
  index_good <- data$Class == "Good"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$Bad[index_bad], predictions$Bad[index_good], curve = TRUE)
  
}
calc_auprc(rangerFit1, testing)
plot(calc_auprc(rangerFit1, testing))

## Feature Importance
names(rangerFit1$modelInfo)

rangerImp <- varImp(rangerFit1)
rangerImp
plot(rangerImp, top=20, cex=.8)


##b3)glmnet ----
library(glmnet)
glmnet_grid <- expand.grid(alpha = c(0, .2, .5),
                           lambda = c(.001, .01, .1)) 

# set seed to run fully reproducible model in parallel mode using caret          
set.seed(828)
glmnet_seeds <- vector(mode = "list", length = 4) # length is = (n_repeats*nresampling)+1
for(i in 1:4) glmnet_seeds[[i]] <- sample.int(n=1000, 9) # ...the number of tuning parameter...
glmnet_seeds[[5]]<-sample.int(1000, 1) # for the last model

## 'number'-fold CV, repeat 'repeats' times
glmnet_fitControl <- trainControl(method = "repeatedcv", 
                                  number = 2, repeats = 2,
                                  seeds = glmnet_seeds,
                                  ## Estimate class probabilities
                                  classProbs = TRUE,
                                  ## Evaluate performance using 
                                  ## the following function
                                  summaryFunction = twoClassSummary)

glmnetFit1 <- train(Class ~ ., data = training, 
                    method = "glmnet", 
                    trControl = glmnet_fitControl,
                    family = "binomial",
                    ## This last option is actually one
                    ## for glmnet() that passes through
                    tuneGrid = glmnet_grid,
                    metric = "ROC")
# plot(gbmFit1, metric = 'Kappa')
glmnetFit1


glmnet_pred <- predict(glmnetFit1, testing, type='raw')
# several manually calculated statitics
# testing sensitivity 
(glmnet_recall <- sum(glmnet_pred=="Bad" & glmnet_pred==testing$Class)/sum(testing$Class=="Bad"))
# testing positive predicted value
(glmnet_precision <- sum(glmnet_pred=="Bad" & glmnet_pred==testing$Class)/sum(glmnet_pred=="Bad") )
# testing specificity
(glmnet_specificity <- sum(glmnet_pred=="Good" & glmnet_pred==testing$Class)/sum(testing$Class=="Good"))

confusionMatrix(table(glmnet_pred, testing$Class))


## precision-recall curve: auprc(area under precision and recall curve)
# model = glmnetFit1
# data = testing
calc_auprc <- function(model, data){
  
  index_bad <- data$Class == "Bad"
  index_good <- data$Class == "Good"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$Bad[index_bad], predictions$Bad[index_good], curve = TRUE)
  
}
calc_auprc(glmnetFit1, testing)
plot(calc_auprc(glmnetFit1, testing))

## Feature Importance


glmnetImp <- varImp(glmnetFit1)
glmnetImp
plot(glmnetImp, top=20, cex=.8)



### b#)Shutdown the cluster ----

stopCluster(cl)




