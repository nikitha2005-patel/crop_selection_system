import joblib
import sklearn
import numpy as np
import pandas as pd
import streamlit as st

model = joblib.load("crop_recommendation_model.pkl")
crop_data = {
    20: {'name': 'Rice', 'image_path': 'https://blog.bigbasket.com/wp-content/uploads/2018/07/hmt-kolam-rice.jpg'},
    11: {'name': 'Maize', 'image_path': 'https://cdn.britannica.com/36/167236-050-BF90337E/Ears-corn.jpg?w=400&h=300&c=crop'},
    3: {'name': 'Chickpea', 'image_path': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgxPC2qqWrxYJjUq4nMVO4BXBI1-vMLv4_cmF1wOGMiWMcSVw_szvlPVkPXzKyjCb96YE&usqp=CAU'},
    9: {'name': 'Kidney Beans', 'image_path': 'https://veggiegardenseeds.com.au/cdn/shop/products/image_d8d5c448-97ba-4a83-886a-024bbbf912ea.jpg?v=1641620451'},
    18: {'name': 'Pigeon Peas', 'image_path': 'https://m.media-amazon.com/images/I/81+2me3gpGL._AC_UF894,1000_QL80_.jpg'},
    13: {'name': 'Moth Beans', 'image_path': 'https://i0.wp.com/amaralfarms.com/wp-content/uploads/2021/11/What_Month_Do_You_Plant_Beans_637727397835005304.png?fit=1000%2C703&ssl=1'},
    14: {'name': 'Mung Bean', 'image_path': 'https://thumbs.dreamstime.com/b/mung-bean-plant-close-up-fruits-scientific-name-vigna-radiata-195861069.jpg'},
    2: {'name': 'Black Gram', 'image_path': 'https://www.asiafarming.com/wp-content/uploads/2018/02/Black-Gram-Growing-and-Cultivation-Practices.-e1523079398684.jpg'},
    10: {'name': 'Lentil', 'image_path': 'https://www.apnikheti.com/upload/crops/7175idea99red_lentil_16x9.jpg'},
    19: {'name': 'Pomegranate', 'image_path': 'https://media.istockphoto.com/id/186548424/photo/pomegranate.jpg?s=612x612&w=0&k=20&c=IM6MPkr4hCp9jsaXMJ5cOsfeLfni31HV3cIqGLjroVQ='},
    1: {'name': 'Banana', 'image_path': 'https://sp-ao.shortpixel.ai/client/to_auto,q_lossy,ret_img,w_600/https://www.optimistdaily.com/wp-content/uploads/zone-9-banana-e1575315807804.jpg'},
    12: {'name': 'Mango', 'image_path': 'https://www.thespruce.com/thmb/wpjmSnh43BwMAe7R3I31yydu_rs=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/grow-mango-seeds-1902625-hero-f222c4c0a3ad4b24a70948ddcec9e5e6.jpg'},
    7: {'name': 'Grapes', 'image_path': 'https://media.istockphoto.com/id/477487589/photo/red-grapes.jpg?s=612x612&w=0&k=20&c=Dg4O8uN-mVQIC6yY6kEP0Bu0zaOvQjb05usguMzxYWY='},
    21: {'name': 'Watermelon', 'image_path': 'https://www.thespruce.com/thmb/fghCcR0Sv_lrSIg1mVWS9U-b_ts=/4002x0/filters:no_upscale():max_bytes(150000):strip_icc()/how-to-grow-watermelons-1403491-hero-2d1ce0752fed4ed599db3ba3b231f8b7.jpg'},
    15: {'name': 'Muskmelon', 'image_path': 'https://plantic.in/pimg/pl-muskmelon-f1-hybrid-mithas/pl-muskmelon-f1-hybrid-mithas1.png'},
    0: {'name': 'Apple', 'image_path': 'https://blog-images-1.pharmeasy.in/blog/production/wp-content/uploads/2022/05/03114105/7-5.jpg'},
    16: {'name': 'Orange', 'image_path': 'https://yellowsquash-prod.s3.ap-south-1.amazonaws.com/media/220/oranges1.jpg'},
    17: {'name': 'Papaya', 'image_path': 'https://plantura.garden/uk/wp-content/uploads/sites/2/2022/07/papaya-trees-fruiting.jpg'},
    4: {'name': 'Coconut', 'image_path': 'https://kjcdn.gumlet.io/media/61498/coconut-farm.jpg'},
    6: {'name': 'Cotton', 'image_path': 'https://rukminim2.flixcart.com/image/850/1000/xif0q/plant-seed/e/j/o/14-cotton-seeds-cotton-seeds-hybrid-cotton-seeds-for-planting-original-imagrnp2khqznk3a.jpeg?q=20'},
    8: {'name': 'Jute', 'image_path': 'https://media.istockphoto.com/id/1411013995/photo/green-jute-plantation-field-raw-jute-plant-texture-background.jpg?s=612x612&w=0&k=20&c=cqacEb83QgCc_CNw8VKEulIUgVl4dc9IDkjZR4JBEw0='},
    5: {'name': 'Coffee', 'image_path': 'https://images.theconversation.com/files/472331/original/file-20220704-23-fbnet4.jpg?ixlib=rb-1.1.0&rect=8%2C420%2C5487%2C2743&q=45&auto=format&w=1356&h=668&fit=crop'}
}
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://www.slowfood.com/wp-content/uploads/2021/12/soil-1024x576.png");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()
animation_css = """
<style>
/* CSS for title with animation */
.title {
    font-size: 20px;
    color: white; /* Black font color */
    text-align: center;
    background-color: black; /* White background color */
    padding: 20px; /* Add padding for spacing */
    animation: myAnimation 10s ease infinite; /* CSS animation */
}

/* Keyframes animation */
@keyframes myAnimation {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
</style>
"""

# Apply the CSS animation to the title
st.markdown(animation_css, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Title
st.markdown('<h1 class="title1">Crop Recommendation System</h1><br>', unsafe_allow_html=True)

# Sidebar with user input
st.sidebar.header("User Input")

# Input features
N = st.sidebar.number_input("Nitrogen (N) Ratio", min_value=0.0, max_value=100.0, value=50.0)
P = st.sidebar.number_input("Phosphorous (P) Ratio", min_value=0.0, max_value=100.0, value=30.0)
K = st.sidebar.number_input("Potassium (K) Ratio", min_value=0.0, max_value=100.0, value=20.0)
temperature = st.sidebar.number_input("Temperature (°C)")
humidity = st.sidebar.number_input("Relative Humidity (%)")
ph = st.sidebar.number_input("pH Value", min_value=0.0, value=7.0)
rainfall = st.sidebar.number_input("Rainfall (mm)")

# Recommendation Button
recommend_button = st.sidebar.button("Recommend Crops")

# Machine learning model (replace this with your actual recommendation model)
def recommend_crops(N, P, K, temperature, humidity, ph, rainfall):
    crop = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    print(crop)
   
    if crop[0] in crop_data:
        crop_info = crop_data[crop[0]]
        return st.markdown(f'<h4 class="title">Recommended Crop is {crop_info["name"]}</h4><br>', unsafe_allow_html=True),st.image(crop_info["image_path"], caption=crop_info["name"])
    else:
        st.write("Crop information not found for the selected crop.")
    # Get crop recommendations
if recommend_button:
    # Get crop recommendations
    recommended_crops = recommend_crops(N, P, K, temperature, humidity, ph, rainfall)


# (Optional) Instructions or information
st.sidebar.markdown("Provide input values for N, P, K, temperature, humidity, pH, and rainfall to get crop recommendations.")
