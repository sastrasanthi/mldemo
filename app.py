"""Interactive binary-classification laboratory for Streamlit Community Cloud."""
import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title='Customer Churn | ML Classification Lab', page_icon='📊', layout='wide')
MODEL_NAMES = ['Logistic Regression','Decision Tree','Random Forest','KNN','SVM','Gradient Boosting']
METRICS = ['Accuracy','Precision','Recall','Specificity','F1','Balanced Accuracy','MCC','ROC AUC','PR AUC','Log Loss','Brier Score']
HIGHER_BETTER = {m: m not in ['Log Loss','Brier Score'] for m in METRICS}

@st.cache_data(show_spinner=False)
def load_csv(raw):
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    if df.empty or df.columns.duplicated().any():
        raise ValueError('The CSV is empty or has duplicate column names.')
    for col in df.select_dtypes(include=['object','string']).columns:
        df[col] = df[col].astype('string').str.strip().replace({'':pd.NA, '?':pd.NA})
        if col.lower() == 'totalcharges':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=145, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return buf.getvalue()

def fig_bar(labels, values, title, ylabel='Count', horizontal=False):
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    if horizontal:
        ax.barh([str(x) for x in labels], values, color='#3978b5')
        ax.set_xlabel(ylabel)
    else:
        ax.bar([str(x) for x in labels], values, color='#3978b5')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)
    ax.set_title(title)
    fig.tight_layout()
    return plot_to_png(fig)

def create_eda(df, target):
    charts = {}
    counts = df[target].astype(str).value_counts()
    charts['Target class distribution'] = fig_bar(counts.index, counts.values, 'Target class distribution')
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing>0].head(20)
    if len(missing):
        charts['Missing values'] = fig_bar(missing.index[::-1], missing.values[::-1], 'Missing values (top 20)', horizontal=True)
    nums = df.select_dtypes(include='number').columns.drop(target, errors='ignore').tolist()
    if nums:
        fig, axes = plt.subplots(1, min(4,len(nums)), figsize=(4*min(4,len(nums)),3.7))
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, nums[:4]):
            ax.hist(df[c].dropna(), bins=25, color='#3978b5', edgecolor='white')
            ax.set_title(c)
        fig.suptitle('Numerical feature distributions')
        fig.tight_layout()
        charts['Numerical distributions'] = plot_to_png(fig)
        if len(nums)>=2:
            corr = df[nums[:15]].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(max(6,len(corr)*0.65),max(5,len(corr)*0.55)))
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha='right')
            ax.set_yticks(range(len(corr)), corr.index)
            fig.colorbar(im, ax=ax, shrink=.7)
            ax.set_title('Numeric feature correlations')
            fig.tight_layout()
            charts['Correlation heatmap'] = plot_to_png(fig)
    cats = [c for c in df.columns if c!=target and c not in nums and df[c].nunique(dropna=True) <= 12]
    if cats:
        fig, axes = plt.subplots(1, min(3,len(cats)), figsize=(5*min(3,len(cats)),4))
        axes = np.atleast_1d(axes)
        for ax,c in zip(axes,cats[:3]):
            ct = pd.crosstab(df[c].fillna('Missing').astype(str), df[target].astype(str), normalize='index')
            ct.plot(kind='bar', stacked=True, ax=ax, legend=False)
            ax.set_title(f'{c} vs target')
            ax.tick_params(axis='x', rotation=50)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper right')
        fig.tight_layout(rect=[0,0,.97,.95])
        charts['Categorical churn patterns'] = plot_to_png(fig)
    return charts

def make_models(seed):
    return {
        'Logistic Regression': LogisticRegression(max_iter=1500, class_weight='balanced',random_state=seed),
        'Decision Tree': DecisionTreeClassifier(max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=120,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'SVM': SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=seed),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,random_state=seed)
    }

def train(df, target, positive, drop_ids, missing, scaling, method, k, test_size, seed, chosen, threshold):
    work = df.dropna(subset=[target]).copy()
    y_raw = work[target].astype(str)
    if y_raw.nunique()!=2:
        raise ValueError('Target must have exactly two classes after removing missing targets.')
    y = (y_raw==str(positive)).astype(int).to_numpy()
    X = work.drop(columns=[target]).copy()
    removed=[]
    if drop_ids:
        removed=[c for c in X if c.lower() in ('customerid','customer_id','id','recordid') or
                 (X[c].nunique(dropna=True)==len(X) and not pd.api.types.is_numeric_dtype(X[c]))]
        X=X.drop(columns=removed)
    if X.shape[1]==0: raise ValueError('No input features remain.')
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):
            converted=pd.to_numeric(X[c],errors='coerce')
            if X[c].notna().sum()>0 and converted.notna().sum()/X[c].notna().sum()>=.95:
                X[c]=converted
    if missing=='Drop rows':
        keep=X.notna().all(axis=1)
        X=X.loc[keep]
        y=y[keep.to_numpy()]
    if len(np.unique(y))!=2 or min(np.bincount(y))<2:
        raise ValueError('At least two records of each target class are needed.')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,stratify=y,random_state=seed)
    numeric=X_train.select_dtypes(include='number').columns.tolist()
    categoric=[c for c in X_train if c not in numeric]
    steps=[]
    if numeric:
        scaler={'Standard':StandardScaler(),'Min-Max':MinMaxScaler(),'None':'passthrough'}[scaling]
        steps.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='mean' if missing=='Mean' else 'median',keep_empty_features=True)),('scale',scaler)]),numeric))
    if categoric:
        steps.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categoric))
    preprocessor=ColumnTransformer(steps,sparse_threshold=0)
    A=preprocessor.fit_transform(X_train)
    B=preprocessor.transform(X_test)
    feature_names=np.asarray(preprocessor.get_feature_names_out(),dtype=str)
    if A.shape[1]==0: raise ValueError('No usable features after preprocessing.')
    selector=None
    if method=='ANOVA F-test':
        selector=SelectKBest(f_classif,k=min(k,A.shape[1]))
    elif method=='Mutual Information':
        selector=SelectKBest(lambda x,y: mutual_info_classif(x,y,random_state=seed),k=min(k,A.shape[1]))
    elif method=='L1 Logistic Regression':
        selector=SelectFromModel(LogisticRegression(penalty='l1',solver='liblinear',class_weight='balanced',max_iter=1500,random_state=seed),threshold='median')
    if selector is not None:
        A=selector.fit_transform(A,y_train)
        B=selector.transform(B)
        feature_names=feature_names[selector.get_support()]
    if A.shape[1]==0: raise ValueError('Feature selection removed every feature. Choose another method.')
    output={}
    errors={}
    for name in chosen:
        try:
            model=make_models(seed)[name]
            model.fit(A,y_train)
            prob=model.predict_proba(B)[:,1]
            pred=(prob>=threshold).astype(int)
            tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
            metrics={
                'Accuracy':accuracy_score(y_test,pred),
                'Precision':precision_score(y_test,pred,zero_division=0),
                'Recall':recall_score(y_test,pred,zero_division=0),
                'Specificity':tn/(tn+fp) if tn+fp else np.nan,
                'F1':f1_score(y_test,pred,zero_division=0),
                'Balanced Accuracy':balanced_accuracy_score(y_test,pred),
                'MCC':matthews_corrcoef(y_test,pred),
                'ROC AUC':roc_auc_score(y_test,prob),
                'PR AUC':average_precision_score(y_test,prob),
                'Log Loss':log_loss(y_test,np.clip(prob,1e-15,1-1e-15),labels=[0,1]),
                'Brier Score':brier_score_loss(y_test,prob)
            }
            output[name]={'metrics':metrics,'cm':(tn,fp,fn,tp),'prob':prob,'pred':pred,'model':model}
        except Exception as exc:
            errors[name]=str(exc)
    if not output: raise ValueError(f'All models failed: {errors}')
    table=pd.DataFrame({name:item['metrics'] for name,item in output.items()}).T
    table.index.name='Model'
    return {'table':table,'models':output,'errors':errors,'y_test':y_test,'positive':str(positive),
            'features':feature_names.tolist(),'removed':removed,'train_size':len(y_train),'test_size':len(y_test),
            'preprocessor':preprocessor,'selector':selector,'threshold':threshold,
            'target':target,'input_features':X.columns.tolist(),'selected_method':method}

def evaluation_graphs(result):
    graphs={}
    table=result['table']
    fig,ax=plt.subplots(figsize=(9,4.5))
    table[['Accuracy','Precision','Recall','F1','ROC AUC']].plot(kind='bar',ax=ax,width=.8)
    ax.set_ylim(0,1.05)
    ax.set_title('Classification metrics by model')
    ax.legend(ncol=3,fontsize=8)
    ax.tick_params(axis='x',rotation=30)
    fig.tight_layout()
    graphs['Model metric comparison']=plot_to_png(fig)
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    for name,item in result['models'].items():
        fpr,tpr,_=roc_curve(result['y_test'],item['prob'])
        ax.plot(fpr,tpr,label=f"{name} (AUC {item['metrics']['ROC AUC']:.3f})")
    ax.plot([0,1],[0,1],'k--',alpha=.4)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curves')
    ax.legend(fontsize=8)
    fig.tight_layout()
    graphs['ROC curves']=plot_to_png(fig)
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    for name,item in result['models'].items():
        precision,recall,_=precision_recall_curve(result['y_test'],item['prob'])
        ax.plot(recall,precision,label=f"{name} (AP {item['metrics']['PR AUC']:.3f})")
    ax.set(xlabel='Recall',ylabel='Precision',title='Precision–Recall curves')
    ax.legend(fontsize=8)
    fig.tight_layout()
    graphs['Precision–Recall curves']=plot_to_png(fig)
    for name,item in result['models'].items():
        tn,fp,fn,tp=item['cm']
        cm=np.array([[tn,fp],[fn,tp]])
        fig,ax=plt.subplots(figsize=(4.8,4))
        ax.imshow(cm,cmap='Blues')
        for (i,j),val in np.ndenumerate(cm):
            ax.text(j,i,str(val),ha='center',va='center',fontsize=14)
        ax.set_xticks([0,1],['Negative','Positive'])
        ax.set_yticks([0,1],['Negative','Positive'])
        ax.set(xlabel='Predicted',ylabel='Actual',title=f'{name} – Confusion matrix')
        fig.tight_layout()
        graphs[f'Confusion matrix – {name}']=plot_to_png(fig)
    return graphs

def choose_best(result,metric):
    series=result['table'][metric].dropna()
    if series.empty: raise ValueError(f'No valid {metric} scores.')
    name=series.idxmax() if HIGHER_BETTER[metric] else series.idxmin()
    return name,float(series.loc[name])

def interpretation(result,best,metric):
    item=result['models'][best]
    tn,fp,fn,tp=item['cm']
    m=item['metrics']
    direction='highest' if HIGHER_BETTER[metric] else 'lowest'
    return (f"{best} is selected because it has the {direction} test-set {metric} "
            f"({m[metric]:.4f}) among the successfully trained models. "
            f"Its accuracy is {m['Accuracy']:.1%}, precision is {m['Precision']:.1%}, "
            f"recall is {m['Recall']:.1%}, F1 is {m['F1']:.1%}, "
            f"and ROC-AUC is {m['ROC AUC']:.3f}. "
            f"At threshold {result['threshold']:.2f}, the model produced {tp} true positives, "
            f"{tn} true negatives, {fp} false positives, and {fn} false negatives. "
            "For churn retention, recall emphasizes identifying at-risk customers; precision limits "
            "unnecessary retention offers. ROC-AUC measures ranking across thresholds, not performance "
            "at a single threshold. Scores are estimates on one holdout test set; selecting the winner "
            "on that same test set may give an optimistic estimate. Validate the final choice on new data.")

def pdf_report(df,result,eda,graphs,metric,chosen_metrics):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=(8.27*inch,11.69*inch),leftMargin=42,rightMargin=42,topMargin=45,bottomMargin=45)
    styles=getSampleStyleSheet()
    story=[]
    def para(s,style='BodyText'):
        story.append(Paragraph(escape(str(s)),styles[style]))
        story.append(Spacer(1,8))
    def heading(s): para(s,'Heading2')
    def add_image(title,data):
        heading(title)
        img=Image(io.BytesIO(data))
        maxw,maxh=7.0*inch,5.1*inch
        factor=min(maxw/img.imageWidth,maxh/img.imageHeight)
        img.drawWidth=img.imageWidth*factor
        img.drawHeight=img.imageHeight*factor
        story.append(img)
        story.append(Spacer(1,12))
    para('Customer Churn – Binary Classification Report','Title')
    para(f'Generated: {datetime.now():%Y-%m-%d %H:%M} | Uploaded records: {len(df):,} | Target: {result["target"]}')
    para(f'Train: {result["train_size"]:,} | Test: {result["test_size"]:,} | Positive class: {result["positive"]} | Threshold: {result["threshold"]:.2f}')
    heading('Methodology')
    para('Stratified train/test split. Imputation, scaling, one-hot encoding and optional feature selection are fitted on the training set only to avoid preprocessing leakage. Six standard binary classifiers are supported.')
    para(f'Feature selection: {result["selected_method"]}; selected encoded features: {len(result["features"])}; removed ID columns: {", ".join(result["removed"]) or "none"}.')
    heading('Dataset overview')
    para(f'Rows: {df.shape[0]:,}; columns: {df.shape[1]}; missing cells: {int(df.isna().sum().sum()):,}.')
    heading('Model comparison')
    visible=[m for m in chosen_metrics if m in result['table']]
    cols=['Model']+visible
    table_data=[cols]+[[str(idx)]+[f'{row[c]:.4f}' for c in visible] for idx,row in result['table'].iterrows()]
    width=(7.1*inch)/len(cols)
    tab=Table(table_data,colWidths=[width]*len(cols),repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCE8F5')),
        ('GRID',(0,0),(-1,-1),.35,colors.grey),('FONTSIZE',(0,0),(-1,-1),6.4),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F9FC')]),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(tab)
    story.append(Spacer(1,15))
    best,score=choose_best(result,metric)
    heading('Decision and interpretation')
    para(interpretation(result,best,metric))
    heading('Metric definitions')
    para('Accuracy=(TP+TN)/N; Precision=TP/(TP+FP); Recall=TP/(TP+FN); Specificity=TN/(TN+FP); F1=2PR/(P+R). ROC-AUC summarizes TPR versus FPR across thresholds. PR AUC here uses average precision. Log loss and Brier score are lower-is-better probability-quality metrics.')
    if result['errors']:
        heading('Model warnings')
        for name,err in result['errors'].items(): para(f'{name}: {err}')
    heading('Exploratory data analysis')
    for title,png in eda.items(): add_image(title,png)
    heading('Model evaluation graphs')
    for title,png in graphs.items(): add_image(title,png)
    heading('Limitations')
    para('This is an educational demonstration. No cross-validation, nested model selection, hyperparameter optimization, or external validation is performed. Performance may vary with split, threshold, dataset composition and data quality. Remove personal identifiers and do not upload confidential customer data to a public app.')
    doc.build(story)
    return buf.getvalue()

st.title('📊 Customer Churn – Classification & Model Comparison')
st.caption('Upload a binary-classification CSV, explore data, preprocess, select features, train six classifiers, compare metrics, and export a PDF report.')
with st.sidebar:
    st.header('1 · Dataset')
    uploaded=st.file_uploader('Upload CSV',type=['csv'])
    st.caption('Use an anonymized dataset; do not upload private customer records to a public deployment.')
if uploaded is None:
    st.info('Upload a CSV file using the left sidebar to begin. The target column must contain exactly two classes.')
    st.stop()
try:
    raw=uploaded.getvalue()
    df=load_csv(raw)
except Exception as exc:
    st.error(f'Could not read CSV: {exc}')
    st.stop()

cols=df.columns.tolist()
suggested=next((c for c in cols if c.lower() in ('churn','target','label','class')),cols[-1])
with st.sidebar:
    target=st.selectbox('Target column',cols,index=cols.index(suggested))
    classes=sorted(df[target].dropna().astype(str).unique().tolist())
    if len(classes)==2:
        default=next((v for v in classes if v.lower() in ('yes','1','true','churn','positive','pneumonia')),classes[-1])
        positive=st.selectbox('Positive class',classes,index=classes.index(default))
    else:
        positive=None
        st.warning('Choose a target with exactly two classes.')
    st.header('2 · Preprocessing')
    drop_ids=st.checkbox('Remove likely identifier columns',True)
    missing=st.selectbox('Missing values',['Median','Mean','Drop rows'])
    scaling=st.selectbox('Numeric normalization',['Standard','Min-Max','None'])
    test_size=st.slider('Test proportion',.15,.40,.25,.05)
    seed=st.number_input('Random seed',min_value=0,max_value=100000,value=42)
    st.header('3 · Feature selection')
    method=st.selectbox('Feature selection',['None','ANOVA F-test','Mutual Information','L1 Logistic Regression'])
    k=st.slider('Top k encoded features (ANOVA / MI)',1,100,20)
    st.header('4 · Models')
    chosen=st.multiselect('Classifiers',MODEL_NAMES,default=MODEL_NAMES)
    c1,c2=st.columns(2)
    with c1: run_all=st.button('▶ Run all',use_container_width=True,type='primary')
    with c2: run_selected=st.button('Run selected',use_container_width=True)
    st.header('5 · Evaluation')
    shown_metrics=st.multiselect('Metrics to display',METRICS,default=METRICS)
    decision_metric=st.selectbox('Choose best model by',METRICS,index=METRICS.index('F1'))
    threshold=st.slider('Classification threshold',.05,.95,.50,.05)

signature=(hash(raw),target,positive,drop_ids,missing,scaling,test_size,seed,method,k,tuple(chosen),threshold)
if run_all or run_selected:
    selected=MODEL_NAMES if run_all else chosen
    if positive is None:
        st.error('The target must have exactly two non-missing classes.')
    elif not selected:
        st.error('Select at least one classifier.')
    else:
        try:
            with st.spinner('Preprocessing and training models…'):
                result=train(df,target,positive,drop_ids,missing,scaling,method,k,test_size,int(seed),selected,threshold)
                graphs=evaluation_graphs(result)
            st.session_state['analysis']=(signature,result,graphs)
            st.success(f'Trained {len(result["models"])} classifier(s) successfully.')
        except Exception as exc:
            st.error(f'Training failed: {exc}')

st.subheader('Dataset overview')
a,b,c,d=st.columns(4)
a.metric('Rows',f'{len(df):,}')
b.metric('Columns',str(df.shape[1]))
c.metric('Missing cells',f'{df.isna().sum().sum():,}')
d.metric('Target classes',str(df[target].nunique(dropna=True)))

eda_tab,prep_tab,model_tab,metric_tab,graph_tab,result_tab,decision_tab=st.tabs([
    'EDA','Preprocessing','Feature selection','Evaluation metrics','Graphs','Results','Decision & PDF'])
with eda_tab:
    st.write('Preview')
    st.dataframe(df.head(25),use_container_width=True)
    st.write('Column types and missing values')
    st.dataframe(pd.DataFrame({'dtype':df.dtypes.astype(str),'missing':df.isna().sum(),'unique':df.nunique(dropna=True)}),use_container_width=True)
    if st.button('📈 Generate EDA graphs'):
        with st.spinner('Generating exploratory charts…'):
            st.session_state['eda']=(hash(raw),target,create_eda(df,target))
    eda_cache=st.session_state.get('eda')
    if eda_cache and eda_cache[:2]==(hash(raw),target):
        for title,png in eda_cache[2].items():
            st.image(png,caption=title,use_container_width=True)
    else:
        st.info('Click Generate EDA graphs to view distributions, missingness, correlations, and category patterns.')
with prep_tab:
    st.markdown('**Preprocessing settings** are in the sidebar. Missing values and numeric scaling are fitted using training data only.')
    st.write(f'Numeric columns: {len(df.select_dtypes(include="number").columns)}; categorical/other columns: {len(df.select_dtypes(exclude="number").columns)}.')
    st.write('One-hot encoding is applied to categorical features; unknown test categories are ignored.')
    st.write('Train/test split is stratified by the selected target.')
with model_tab:
    st.write(f'Feature-selection method: **{method}**')
    st.write('ANOVA and mutual information select the top k one-hot-encoded features. L1 logistic regression selects features using sparse coefficients.')
    st.write('The selector is fitted to training data only, avoiding feature-selection leakage.')
    analysis=st.session_state.get('analysis')
    if analysis and analysis[0]==signature:
        result=analysis[1]
        st.metric('Selected encoded features',len(result['features']))
        st.dataframe(pd.DataFrame({'Selected feature':result['features']}),use_container_width=True)
with metric_tab:
    st.markdown('**Classification metrics** are calculated on the held-out test set. Select the metrics to display in the sidebar.')
    st.write('Accuracy, precision, recall, specificity, F1, balanced accuracy, MCC, ROC-AUC, average precision (PR-AUC), log loss and Brier score are supported.')
    st.caption('ROC-AUC and average precision use predicted probabilities; threshold-dependent metrics use the chosen classification threshold.')
with graph_tab:
    analysis=st.session_state.get('analysis')
    if analysis and analysis[0]==signature:
        available=analysis[2]
        choices=st.multiselect('Choose graphs to display',list(available),default=['Model metric comparison','ROC curves','Precision–Recall curves'])
        if st.button('📉 Show selected graphs'):
            for title in choices: st.image(available[title],caption=title,use_container_width=True)
    else:
        st.info('Run the models to enable evaluation graphs.')
with result_tab:
    analysis=st.session_state.get('analysis')
    if analysis and analysis[0]==signature:
        result=analysis[1]
        st.subheader('Comparison table')
        st.dataframe(result['table'][shown_metrics].style.format('{:.4f}'),use_container_width=True)
        st.download_button('⬇ Download comparison CSV',result['table'].to_csv().encode(),'model_comparison.csv','text/csv')
        if result['errors']: st.warning(f'Models that failed: {result["errors"]}')
    else:
        st.info('Click Run all or Run selected to display the comparison table.')
with decision_tab:
    analysis=st.session_state.get('analysis')
    if analysis and analysis[0]==signature:
        result,graphs=analysis[1],analysis[2]
        best,score=choose_best(result,decision_metric)
        if st.button('🏆 Show best-model decision'):
            st.success(f'Best model by {decision_metric}: {best} ({score:.4f})')
            st.write(interpretation(result,best,decision_metric))
        if st.button('📄 Generate full PDF report'):
            try:
                eda_cache=st.session_state.get('eda')
                eda=eda_cache[2] if eda_cache and eda_cache[:2]==(hash(raw),target) else create_eda(df,target)
                with st.spinner('Creating PDF report with tables and charts…'):
                    pdf=pdf_report(df,result,eda,graphs,decision_metric,shown_metrics or METRICS)
                st.session_state['pdf']=(signature,decision_metric,tuple(shown_metrics),pdf)
                st.success('PDF report created.')
            except Exception as exc:
                st.error(f'PDF generation failed: {exc}')
        pdf_cache=st.session_state.get('pdf')
        if pdf_cache and pdf_cache[:3]==(signature,decision_metric,tuple(shown_metrics)):
            st.download_button('⬇ Download complete PDF report',pdf_cache[3],file_name='customer_churn_analysis_report.pdf',mime='application/pdf')
    else:
        st.info('Train models first. Then display the best-model decision or generate the downloadable report.')

analysis=st.session_state.get('analysis')
if analysis and analysis[0]!=signature:
    st.warning('Settings have changed since the last training run. Click Run all or Run selected to refresh results.')
