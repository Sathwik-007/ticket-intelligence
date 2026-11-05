import streamlit as st
import requests

# Configure page settings
st.set_page_config(page_title="Ticket Intelligence", page_icon="🎫", layout="centered")

# Inject custom CSS for modern ticket cards
st.markdown("""
<style>
    .ticket-card {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #0f172a;
    }
    .distance-badge {
        color: #3b82f6;
        font-weight: 600;
        font-size: 0.85rem;
        background: #e0e7ff;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎫 Ticket Intelligence Router")
st.markdown("Semantic Routing & Nearest Neighbor Search")

user_input = st.text_area("Customer Ticket Input:", height=120, placeholder="E.g., My order is delayed, when will it arrive?")

if st.button("Route Ticket", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a ticket to route.")
    else:
        with st.spinner("Analyzing semantic intent..."):
            try:
                # Call the decoupled FastAPI backend
                response = requests.post(
                    "http://localhost:8000/route-ticket",
                    json={"text": user_input}
                )
                response.raise_for_status()
                data = response.json()
                
                st.divider()
                
                # Display cluster assignment metrics
                col1, col2 = st.columns(2)
                col1.metric("Assigned Cluster", data["cluster_id"])
                col2.metric("Routing Confidence", f"{data['confidence'] * 100:.1f}%")
                
                st.subheader("🔗 Top 5 Historical Matches")
                
                # Render results using the custom CSS cards
                for ticket in data["similar_tickets"]:
                    st.markdown(f"""
                    <div class="ticket-card">
                        <p style="margin-bottom: 0.75rem; font-size: 1.05rem;">"{ticket['text']}"</p>
                        <span class="distance-badge">L2 Distance: {ticket['distance']:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Backend connection failed. Ensure FastAPI is running on port 8000. Error: {e}")