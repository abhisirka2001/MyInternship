
# Function to extract the sentiment from the feedback of egg customers
#pip install openai==0.28
import pandas as pd
import openai

openai.api_key = '' # Generate your own openai api key

df = pd.read_excel("C:\\Users\\User\\Downloads\\Eggoz\\Sentiment_Analysis\\Data\\Sentiment Analysis.xlsx") # read the excel file containing sentiments
df=df.dropna() # run this command to remove any NA rows

def analyze_gpt35(text):
    messages = [
        {"role": "system", "content": """You are trained to analyze and detect the sentiment of given feedbacks from egg consumers.You are doing it for the Eggoz brand. 
                                        If you're unsure of an answer, you can say "not sure" and recommend to review manually."""},
        {"role": "user", "content": f"""Analyze the following product review and determine if the sentiment is: Good or Bad or Expensive or Normal or OtherBrand. 
                                        Return answer in single word as either Good or Bad or Expensive or Normal or OtherBrand: {text}"""}
        ]
   
    response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages, 
                      max_tokens=1, 
                      n=1, 
                      stop=None, 
                      temperature=0)

    response_text = response.choices[0].message.content.strip().lower()

    return response_text

df["Predicted"]=df["Feedback"].apply(analyze_gpt35)
df.to_csv("Predicted_sentiment.csv",index=False)