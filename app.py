import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc

import pandas as pd

# from flask_caching import Cache

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

###### Define app & cache ##########
# VAPOR, LUX, QUARTZ
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

# IMPORTANT: this line is needed for deployment
server = app.server 

# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory'
# })

####################################

###### load models ##################
# cache if possible?
# I tried caching with local filesystem, computer ran out of space lol
# TIMEOUT = 60

# @cache.memoize(timeout=TIMEOUT)
# def load_models():
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#     model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
#     return tokenizer, model

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

#####################################

###### Chat bot functions ###########
def textbox(text, box="AI", name="Philippe"):
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        # thumbnail = html.Img(
        #     src=app.get_asset_url("Philippe.jpg"),
        #     style={
        #         "border-radius": 50,
        #         "height": 36,
        #         "margin-right": 5,
        #         "float": "left",
        #     },
        # )
        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)

        return html.Div([textbox])

    else:
        raise ValueError("Incorrect option for `box`.")


# Define Layout
conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

controls = dbc.InputGroup(
    children=[
        dbc.Input(id="user-input", placeholder="Write to the chatbot...", type="text", style={"color":"black", "backgroundColor": "white"}),
        dbc.Button("Submit", id="submit")
    ]
)

#####################################




navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Hugging Face Demo",
    brand_href="#",
    color="primary",
    dark=True,
)

# description = """
# TODO: fill in description
# """

# app.layout = [navbar, dbc.Container()]
app.layout = html.Div([navbar, 
    
    # first container
    dbc.Container([
        # r1
        dbc.Row([
            # r1 c1
            dbc.Col([
                
                html.H1(
                    children='Chat Bot Demo',
                    style={
                        'textAlign': 'center',
                    }
                ),
                
                html.Div(
                    "This is a demo I made to try out the dialogGPT model from Hugging Face's transformers library. Enter text below and start chatting!"
                )
                
            ], class_name='divBorder'),
            
        
        ], className="g-0",),
        

        

    
    
    dcc.Store(id="store-conversation", data=""),
    conversation,
    controls,
    dbc.Spinner(html.Div(id="loading-component"))
    
        ])
    
])

# dbc.Container([navbar,  
#                 dbc.Container(
                    
#                 )                     
# ], fluid = True)

############### Callback


@app.callback(
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history):
    print("hi")
    return [
        textbox(x, box="user") if i % 2 == 0 else textbox(x, box="AI")
        for i, x in enumerate(chat_history.split("<split>")[:-1])
    ]


# clears input after user submits text
@app.callback(
    Output("user-input", "value"),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input(n_clicks, n_submit):
    return ""


# @app.callback(
#     [Output("store-conversation", "data"), Output("loading-component", "children")],
#     [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
#     [State("user-input", "value"), State("store-conversation", "data")],
# )
# def run_chatbot_2(n_clicks, n_submit, user_input, chat_history):
#     if n_clicks == 0 and n_submit is None:
#         return "", None
    
#     model_output = "busy, will talk later"
#     chat_history += model_output
#     return chat_history, None
    
@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0 and n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None
    
    name = "Pete"
    
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history == "":
        chat_history_ids = torch.empty(size=(1,1), dtype=torch.int64) #torch.empty(1,1)
        
    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if n_submit else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # First add the user input to the chat history
    chat_history += f"You: {user_input}<split>{name}: "

    model_input = chat_history.replace("<split>", "\n")
    
    # response = openai.Completion.create(
    #     engine="davinci",
    #     prompt=model_input,
    #     max_tokens=250,
    #     stop=["You:"],
    #     temperature=0.9,
    # )
    # model_output = response.choices[0].text.strip()
    
    model_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(model_output)

    chat_history += f"{model_output}<split>"

    return chat_history, None


if __name__=='__main__':
    app.run_server(debug=True, port=8005)

