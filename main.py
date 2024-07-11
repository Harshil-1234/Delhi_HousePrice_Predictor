from flask import Flask,render_template,request
import pandas as pd
import pickle
import json
import numpy as np

__locations = None
__data_columns = None


app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("Prediction_Model.pkl",'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET'])
def prediction():
    localities = sorted(data['Locality'].unique())
    # status = sorted(data['Status'].unique())
    # transaction = sorted(data['Transaction'].unique())
    # return render_template('prediction.html',localities=localities,status=status,transaction=transaction)
    return render_template('prediction.html',localities=localities)

@app.route('/prediction', methods=['POST'])
def predict():

    # per_sqft = float(request.form.get('per_sqft'))
    locality = request.form.get('locality')
    bhk = float(request.form.get('bhk'))
    bathroom = float(request.form.get('bathroom'))
    area = float(request.form.get('area'))

    #to convert square gazz to sqft area
    area = (area * (8.91359))

    # Checking for some edge cases
    if((((float)(area) / bhk)<10) or (((float)(area) / bathroom)<10)):
        return "Houses of the given requirements are not available in this locality"
    
    # status = request.form.get('status')
    # transaction = request.form.get('trasnaction')

    # print(locality,bhk,bathroom,area,status,transaction,per_sqft)
    # print(locality,bhk,bathroom,area)

    # input = pd.DataFrame([[area,bhk,bathroom,locality,status,transaction,per_sqft]],columns=['Area','BHK','Bathroom','Locality','Status','Transaction','Per_Sqft'])
    # input = pd.DataFrame([[area,bhk,bathroom,locality]],columns=['Area','BHK','Bathroom','Locality'])

    global  __data_columns
    global __locations

    with open('columns.json', "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    try:
        loc_index = __data_columns.index(locality.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = area
    x[1] = bathroom
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    result=pipe.predict([x])[0]

    if(result < 0 ):
        return "Houses of the given requirements are not available in this locality"

    result = str(np.round(result,2))

    result = format_number(result)

    return result

def format_number(number_string):
    # Convert the string to a float
    number = float(number_string)
    # Split the number into integer and decimal parts
    integer_part, decimal_part = str(number).split('.')
    # Reverse the integer part for easier processing
    integer_part = integer_part[::-1]
    
    # Group digits in Indian format
    grouped_digits = []
    for i in range(len(integer_part)):
        if i > 2 and (i - 1) % 2 == 0:
            grouped_digits.append(',')
        grouped_digits.append(integer_part[i])
    
    # Reverse again to get the correct order
    formatted_integer_part = ''.join(grouped_digits)[::-1]
    
    # Combine integer part and decimal part
    formatted_number = formatted_integer_part + '.' + decimal_part
    return formatted_number

if __name__ == "__main__":
    app.run(port=8000,debug=True)