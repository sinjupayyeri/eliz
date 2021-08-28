from pycaret.classification import load_model,predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('Final_model')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions=predictions_df['Label'][0]
    return predictions


def run():
    from PIL import Image
    image = Image.open('pexels-pixabay-415779.jpg')
    image_office = Image.open('pexels-pixabay-40568.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox('How would you like to predict?',('Online','Batch'))

    st.sidebar.info('This app is created to predict if a person will suffer from a heart attck or not')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)


    st.title('Heart Attack Analysis')

    if add_selectbox == 'Online':
        age = st.number_input('Age', min_value=1, max_value=100,value=5)
        sex = st.selectbox('Sex',['Male','Female'])
        cp = st.number_input('ChestPain Type', min_value=0, max_value=3,value=1)
        trtbps = st.number_input('Resting Blood Pressure', min_value=94, max_value=200,value=120)
        chol = st.number_input('Cholestrol', min_value=120, max_value=600,value=140)
        fbs = st.number_input('Fasting Blood Sugar', min_value=0, max_value=1,value=0)
        restecg = st.number_input('Resting ECG', min_value=0, max_value=2,value=0)
        thalachh = st.number_input('Maximum HRA',min_value=70, max_value=250,value=120)
        oldpeak = st.number_input('Previous Peak',min_value=0, max_value=7,value=1)
        slp = st.number_input('Slope',min_value=0, max_value=2,value=0)
        caa = st.number_input('Major Blood Vessels',min_value=0, max_value=3,value=1)

        output = ""

        input_dict ={'age':age,'sex':sex,'cp':cp,'trtbps':trtbps,
                     'chol':chol,'fbs':fbs,'restecg':restecg,'thalachh':thalachh,'oldpeak':oldpeak,
                     'slp':slp,'caa':caa}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
        
        
   
def main():
    run()



if __name__ == "__main__":
    main()
