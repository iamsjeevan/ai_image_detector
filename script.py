import streamlit as st
import requests
import json
import io # Needed to handle the uploaded file bytes correctly

# --- Configuration ---
st.set_page_config(page_title="AI Image Detector", layout="centered")
st.title("🤖 AI-Generated Image Detector")
st.write("Upload an image to check the probability of it being AI-generated using the machine learning model.")

# --- API Credentials Handling (using Streamlit Secrets) ---
try:
    API_USER = st.secrets["sightengine"]["api_user"]
    API_SECRET = st.secrets["sightengine"]["api_secret"]
except KeyError:
    st.error("🚨 machine learning model credentials not found! Please configure them in Streamlit Secrets.")
    st.info("Create a file named `.streamlit/secrets.toml` with the following content:\n```toml\n[sightengine]\napi_user = \"YOUR_API_USER\"\napi_secret = \"YOUR_API_SECRET\"\n```")
    st.stop()
except FileNotFoundError:
     st.error("🚨 `.streamlit/secrets.toml` file not found.")
     st.info("Create a file named `.streamlit/secrets.toml` with the following content:\n```toml\n[sightengine]\napi_user = \"YOUR_API_USER\"\napi_secret = \"YOUR_API_SECRET\"\n```")
     st.stop()


# --- Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True) # <--- CORRECTED HERE

    # Prepare for API call
    st.write("Analyzing image...")

    params = {
      'models': 'genai',
      'api_user': API_USER,
      'api_secret': API_SECRET
    }

    file_bytes = uploaded_file.getvalue()
    files = {'media': (uploaded_file.name, file_bytes, uploaded_file.type)}


    # --- API Call and Result Display ---
    try:
        with st.spinner('🧠 Contacting machine learning model...'):
            response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params, timeout=20)
            response.raise_for_status()

        output = response.json()

        if output.get("status") == "success":
            ai_prob = output.get("type", {}).get("ai_generated", None)

            if ai_prob is not None:
                st.success("✅ Analysis Complete!")
                st.metric(label="AI Generation Probability", value=f"{ai_prob:.2%}")

                if ai_prob > 0.75:
                    st.warning("High probability of being AI-generated.")
                elif ai_prob > 0.5:
                    st.info("Moderate probability of being AI-generated.")
                elif ai_prob > 0.1:
                     st.info("Low probability of being AI-generated.")
                else:
                    st.success("Very low probability of being AI-generated (likely human-created).")
            else:
                st.error("❌ Could not extract AI probability from the API response.")
                st.json(output)
        else:
            st.error(f"❌ API Error: {output.get('error', {}).get('message', 'Unknown error')}")
            st.json(output)

    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Network Error: Failed to connect to machine learning model. {e}")
    except json.JSONDecodeError:
        st.error("🚨 Error: Could not decode the response from the API (invalid JSON).")
        st.text(response.text)
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred: {e}")

else:
    st.info("☝️ Upload an image file (jpg, jpeg, png) to start the analysis.")