#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms #import cross_val_scores
import csv
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer
from sklearn_porter import Porter


# data_file = '/mnt/drive/work/sample_rows.csv'
data_file = '/home/shitong.zhu/Desktop/unique_features_data.csv'
events = ['NetworkLinkRequest', 'NetworkScriptRequest', 'NetworkImageRequest', 'NetworkIframeRequest', 'NetworkXMLHTTPRequest', 'NetworkVideoRequest']
tags = ['UNKNOWN','UNCOMMON','HEAD','BODY','DIV','VIDEO','FORM','HTML','A','SPAN','SECTION','LI','PICTURE','UL','MAIN','HEADER','TD','FOOTER','P','FIGURE','NAV','CENTER','B','ASIDE','IFRAME','DT','INS','MAINCONTENT','H4','SCRIPT','LEFT','STRONG','ARTICLE','URSCRUB','FONT','I','H1','LABEL','H2','PRE','BUTTON','head','body','NOINDEX','TR','A-IMG','TABLE','NOSCRIPT','SMALL','ISGOOGLEREMARKETING','BODYCLASS=HOME','TH','TERMS-BUTTON','W7-COMPONENT','NOLAYER','MYBASE','DL','HOME-PAGE','TEXT','FRAGMENT','CUSTOMHTML','RA','TBODY','LINK','BASE','META','STYLE','IMG','TITLE','SOURCE','INPUT','COOKIEJAR','BR','CODE','FB:LIKE','CLOUDFLARE-APP','svg','H3','CANVAS','AUDIO','H5','TWC-NON-EXPANDABLE-WIDGET','APP-ROOT','VT-APP','ENVIRONMENT-OVERLAY','APP','GTM','ONL-ROOT','COOKIE-CONSENT','PHOENIX-SCRIPT','PHOENIX-PAGE','OBJECT','IAS-AD','FBTRASH','BLOCKQUOTE','YMAPS','APM_DO_NOT_TOUCH','WB:FOLLOW-BUTTON','script','UI-VIEW','ICON','CUSTOM-STYLE','DASHI-SERVICE-WORKER','DASHI-ANALYTICS-TAG','ABBR','AMP-GEO','ALPS-ACCOUNT-PANEL','IMIMXXXYYY','AMP-CONSENT','W-DIV','CFC-APP-ROOT','DASHI-LINK-SCANNER']

# tag_1 = ['HEAD','BODY','DIV','VIDEO','FORM','HTML','IE:HOMEPAGE','A','SPAN','SECTION','LI','PICTURE','UL','MAIN','HEADER','TD','FOOTER','P','FIGURE','NAV','CENTER','B','ASIDE','IFRAME','DT','INS','MAINCONTENT','H4','SCRIPT','LEFT','STRONG','ARTICLE','URSCRUB','FONT','I','ESI:TEXT','H1','YT-IMG-SHADOW','FJTIGNOREURL','FIELDSET','CNX','DD','LABEL','H2','PRE','BUTTON','head','body','NOINDEX','TR','A-IMG','EM','OC-COMPONENT','TMPL_IF','TABLE','NOSCRIPT','YATAG','HINETWORK','LINKÂ HREF=','METAÃ','SMALL','ISGOOGLEREMARKETING','BODYCLASS=HOME','TH','OLANG','C:IF','Pæ¥æ¬éç¨®ä¸ç­çé­çæ°ææè¿å çºèç¨±æ¥æ¬25å¹´ä¾çæå¼·é¢±é¢¨ãçå­ãä¸é¸è®ä¸å°å¬å¸ç¼åºï¼å¸°å®å½ä»¤ï¼','TERMS-BUTTON','W7-COMPONENT','METAÂ NAME=CXENSEPARSE:URLÂ CONTENT=HTTP:','MKT-HERO-MARQUEE','NOLAYER','METAâ','MYBASE','DL','HOME-PAGE','CFC-IMAGE','TEXT','WAINCLUDE','METAÂ PROPERTY=FB:PAGES','FRAGMENT','GWC:HIT','CUSTOMHTML','ESI:INCLUDE','RA','TBODY','HTMLÂ XMLNS=HTTPS:','VNN']
# tag_2 = ['UNKNOWN','UNCOMMON','SCRIPT','LINK','IFRAME','BASE','META','DIV','STYLE','BODY','IMG','NOSCRIPT','A','HEADER','VIDEO','TITLE','RMFNLJMLWDDURTPYLAMWH','FOOTER','SOURCE','MAIN','INPUT','SPAN','SECTION','COOKIEJAR','BR','URSCRUB','FORM','P','UL','CODE','FB:LIKE','NAV','LI','CLOUDFLARE-APP','svg','H2','H3','CANVAS','AUDIO','H5','TABLE','TWC-NON-EXPANDABLE-WIDGET','H4','CENTER','INS','APP-ROOT','HEAD','VT-APP','ENVIRONMENT-OVERLAY','I','APP','GTM','ONL-ROOT','MYBASE','COOKIE-CONSENT','PHOENIX-SCRIPT','PHOENIX-PAGE','OBJECT','IAS-AD','BUTTON','EM','FBTRASH','BLOCKQUOTE']
# tag_3 = ['UNKNOWN','HEAD','DIV','BODY','SPAN','UL','FORM','LI','ASIDE','P','SECTION','HTML','TBODY','A','ESI:TEXT','H1','FJTIGNOREURL','CLOUDFLARE-APP','CNX','B','FOOTER','BUTTON','INS','CENTER','YATAG','SCRIPT','NOINDEX','TD','HEADER','head','body','HINETWORK','MAIN','ARTICLE','YMAPS','IFRAME','NOSCRIPT','FONT','FIELDSET','H3','FIGURE','ESI:INCLUDE','Pæ¥æ¬éç¨®ä¸ç­çé­çæ°ææè¿å çºèç¨±æ¥æ¬25å¹´ä¾çæå¼·é¢±é¢¨ãçå­ãä¸é¸è®ä¸å°å¬å¸ç¼åºï¼å¸°å®å½ä»¤ï¼','DT','MYBASE','PHOENIX-SCRIPT','APM_DO_NOT_TOUCH','SMALL','FRAGMENT','WB:FOLLOW-BUTTON','TH','NAV','HTMLÂ XMLNS=HTTPS:']
# tag_4 = ['LI','DL','P','ASIDE','INS','HEAD','script','UI-VIEW','ICON','CUSTOM-STYLE','BASE','svg','I','NOINDEX','DASHI-SERVICE-WORKER','DASHI-ANALYTICS-TAG','ABBR','CANVAS','AMP-GEO','H2','ESI:INCLUDE','ALPS-ACCOUNT-PANEL','IMIMXXXYYY','NAV','PHOENIX-SCRIPT','AMP-CONSENT','W-DIV','CFC-APP-ROOT','URSCRUB','CENTER','H3','DASHI-LINK-SCANNER','H4']


def transform_row(row):
    row[13] = events.index(row[13])

    if row[23] in tags:
        row[23] = tags.index(row[23])
    elif row[23].strip() == '':
        row[23] = 0
    else:
        row[23] = 1
    
    if row[26] in tags:
        row[26] = tags.index(row[26])
    elif row[26].strip() == '':
        row[26] = 0
    else:
        row[26] = 1
    
    if row[42] in tags:
        row[42] = tags.index(row[42])
    elif row[42].strip() == '':
        row[42] = 0
    else:
        row[42] = 1
    
    if row[45] in tags:
        row[45] = tags.index(row[45])
    elif row[45].strip() == '':
        row[45] = 0
    else:
        row[45] = 1

    row[4] = round(float(row[4]), 3)
    row[5] = round(float(row[5]), 3)
    row[10] = round(float(row[10]), 3)
    row[32] = round(float(row[32]), 3)
    row[51] = round(float(row[51]), 3)

    return row

def cv_confusion_matrix(clf, X, y, folds=10):    
    skf = StratifiedKFold(n_splits=folds)
    cv_iter = skf.split(X, y)
    cms = []
    #print X
    for train, test in cv_iter:
 
        clf.fit(X[train], y[train])
        res = clf.predict(X[test])
        cm = confusion_matrix(y[test], res, labels=clf.classes_)        
        cms.append(cm)
    print clf.classes_
    return np.sum(np.array(cms), axis=0)

to_exclude = {0,1,9,31,50}
TestFileCsvReader = csv.reader(open(data_file,'rb'), delimiter = ',')
testdata = []
labels = []
TestdataIds = []
next(TestFileCsvReader)
for row in TestFileCsvReader:    
    
    row = transform_row(row)
    # d = [i for i in row[2:-1]]
    # print row[2:-1]
    d = [element for i, element in enumerate(row[:-1]) if i not in to_exclude]
    # print d
    testdata.append(np.array(d))
    labels.append(row[-1])

    # print np.array(d)
    
    # TestdataIds.append(row[0])
    # AllApps[row[0]] = 1

print len(testdata)
clf = RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 1, criterion = "entropy") # n_estimators is numTree. max_features is numFeatures
clf.fit(np.asarray(testdata),np.asarray(labels))
porter = Porter(clf, language='c')
output = porter.export(embed_data=True)

# print output

scores = cv_confusion_matrix(clf,np.asarray(testdata),np.asarray(labels),10)
print scores
tp = scores[0][0]
tn = scores[1][1]
fp = scores[1][0]
fn = scores[0][1]
 
accuracy = round(((tp+tn)*1.0/(tp+tn+fp+fn) * 1.0)*100,2)
FPR = round(((fp)*1.0/(tn+fp)*1.0)*100,2)
Recall = round(((tp)*1.0/(tp+fn)*1.0)*100,2)
precesion = round(((tp)*1.0/(tp+fp)*1.0)*100,2)

print "ACCURACY:" + str(accuracy)
print "FPR:" + str(FPR)
print "Recall:" + str(Recall)
print "Precision:" + str(precesion)
