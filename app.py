
# ============================================================
# HOTEL BOOKING CANCELLATION PREDICTION
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hotel Cancellation Risk Predictor",
    page_icon="🏨",
    layout="wide"
)


# ============================================================
# 2. LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("data/output/best_rf_model.pkl")
    preprocessor = joblib.load("data/output/preprocessor.pkl")

    return model, preprocessor


model, preprocessor = load_artifacts()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_categorical_options(preprocessor, column_name):

    """
    Get categories learned by OneHotEncoder
    during model training.
    """

    transformer = preprocessor.named_transformers_["cat"]

    categorical_columns = list(
        preprocessor.transformers_[1][2]
    )

    column_index = categorical_columns.index(column_name)

    return list(
        transformer.categories_[column_index]
    )


def create_features(
    hotel,
    lead_time,
    arrival_date_year,
    arrival_date_month,
    arrival_date_week_number,
    arrival_date_day_of_month,
    stays_in_weekend_nights,
    stays_in_week_nights,
    adults,
    children,
    babies,
    meal,
    country,
    market_segment,
    distribution_channel,
    is_repeated_guest,
    previous_cancellations,
    previous_bookings_not_canceled,
    reserved_room_type,
    booking_changes,
    deposit_type,
    agent,
    days_in_waiting_list,
    customer_type,
    adr,
    required_car_parking_spaces,
    total_of_special_requests,
    company_missing
):

    # --------------------------------------------------------
    # Total Guests
    # --------------------------------------------------------

    total_guests = (
        adults +
        children +
        babies
    )

    zero_guest_booking = int(
        total_guests == 0
    )


    # --------------------------------------------------------
    # Total Stay
    # --------------------------------------------------------

    total_stay_nights = (
        stays_in_weekend_nights +
        stays_in_week_nights
    )

    zero_stay_booking = int(
        total_stay_nights == 0
    )


    # --------------------------------------------------------
    # Previous Booking Features
    # --------------------------------------------------------

    total_previous_bookings = (
        previous_cancellations +
        previous_bookings_not_canceled
    )

    if total_previous_bookings > 0:

        previous_cancellation_rate = (
            previous_cancellations /
            total_previous_bookings
        )

    else:

        previous_cancellation_rate = 0


    has_previous_booking = int(
        total_previous_bookings > 0
    )


    # --------------------------------------------------------
    # Lead Time Group
    # --------------------------------------------------------

    if lead_time <= 7:

        lead_time_group = "0-7"

    elif lead_time <= 30:

        lead_time_group = "8-30"

    elif lead_time <= 90:

        lead_time_group = "31-90"

    elif lead_time <= 180:

        lead_time_group = "91-180"

    elif lead_time <= 365:

        lead_time_group = "181-365"

    else:

        lead_time_group = "365+"


    # --------------------------------------------------------
    # Special Request
    # --------------------------------------------------------

    has_special_request = int(
        total_of_special_requests > 0
    )


    # --------------------------------------------------------
    # Booking Changes
    # --------------------------------------------------------

    has_booking_changes = int(
        booking_changes > 0
    )


    # --------------------------------------------------------
    # Waiting List
    # --------------------------------------------------------

    was_on_waiting_list = int(
        days_in_waiting_list > 0
    )


    # --------------------------------------------------------
    # Parking
    # --------------------------------------------------------

    requires_parking = int(
        required_car_parking_spaces > 0
    )


    # --------------------------------------------------------
    # Children / Babies
    # --------------------------------------------------------

    has_children_or_babies = int(
        (children > 0) or
        (babies > 0)
    )


    # --------------------------------------------------------
    # ADR Per Guest
    # --------------------------------------------------------

    if total_guests > 0:

        adr_per_guest = (
            adr / total_guests
        )

    else:

        adr_per_guest = 0


    # --------------------------------------------------------
    # Weekend Stay
    # --------------------------------------------------------

    has_weekend_stay = int(
        stays_in_weekend_nights > 0
    )


    # --------------------------------------------------------
    # Month Number
    # --------------------------------------------------------

    month_mapping = {

        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12

    }

    arrival_month_num = month_mapping[
        arrival_date_month
    ]


    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    if arrival_month_num in [12, 1, 2]:

        arrival_season = "Winter"

    elif arrival_month_num in [3, 4, 5]:

        arrival_season = "Spring"

    elif arrival_month_num in [6, 7, 8]:

        arrival_season = "Summer"

    else:

        arrival_season = "Autumn"


    # --------------------------------------------------------
    # Create Final DataFrame
    # --------------------------------------------------------

    data = {

        "hotel": hotel,

        "lead_time": lead_time,

        "arrival_date_year":
            arrival_date_year,

        "arrival_date_month":
            arrival_date_month,

        "arrival_date_week_number":
            arrival_date_week_number,

        "arrival_date_day_of_month":
            arrival_date_day_of_month,

        "stays_in_weekend_nights":
            stays_in_weekend_nights,

        "stays_in_week_nights":
            stays_in_week_nights,

        "adults": adults,

        "children": children,

        "babies": babies,

        "meal": meal,

        "country": country,

        "market_segment":
            market_segment,

        "distribution_channel":
            distribution_channel,

        "is_repeated_guest":
            is_repeated_guest,

        "previous_cancellations":
            previous_cancellations,

        "previous_bookings_not_canceled":
            previous_bookings_not_canceled,

        "reserved_room_type":
            reserved_room_type,

        "booking_changes":
            booking_changes,

        "deposit_type":
            deposit_type,

        "agent": agent,

        "days_in_waiting_list":
            days_in_waiting_list,

        "customer_type":
            customer_type,

        "adr": adr,

        "required_car_parking_spaces":
            required_car_parking_spaces,

        "total_of_special_requests":
            total_of_special_requests,

        "company_missing":
            company_missing,

        # Engineered features
        "total_guests":
            total_guests,

        "zero_guest_booking":
            zero_guest_booking,

        "total_stay_nights":
            total_stay_nights,

        "zero_stay_booking":
            zero_stay_booking,

        "total_previous_bookings":
            total_previous_bookings,

        "previous_cancellation_rate":
            previous_cancellation_rate,

        "has_previous_booking":
            has_previous_booking,

        "lead_time_group":
            lead_time_group,

        "has_special_request":
            has_special_request,

        "has_booking_changes":
            has_booking_changes,

        "was_on_waiting_list":
            was_on_waiting_list,

        "requires_parking":
            requires_parking,

        "has_children_or_babies":
            has_children_or_babies,

        "adr_per_guest":
            adr_per_guest,

        "has_weekend_stay":
            has_weekend_stay,

        "arrival_month_num":
            arrival_month_num,

        "arrival_season":
            arrival_season
    }


    return pd.DataFrame([data])


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

st.title("🏨 Hotel Booking Cancellation Risk Predictor")

st.write(
    """
    Predict the probability that a hotel booking will be cancelled
    using the trained Random Forest machine-learning model.
    """
)

st.divider()


# ============================================================
# 5. GET CATEGORIES FROM TRAINED PREPROCESSOR
# ============================================================

hotel_options = get_categorical_options(
    preprocessor,
    "hotel"
)

month_options = get_categorical_options(
    preprocessor,
    "arrival_date_month"
)

meal_options = get_categorical_options(
    preprocessor,
    "meal"
)

country_options = get_categorical_options(
    preprocessor,
    "country"
)

market_segment_options = get_categorical_options(
    preprocessor,
    "market_segment"
)

distribution_channel_options = get_categorical_options(
    preprocessor,
    "distribution_channel"
)

room_options = get_categorical_options(
    preprocessor,
    "reserved_room_type"
)

deposit_options = get_categorical_options(
    preprocessor,
    "deposit_type"
)

agent_options = get_categorical_options(
    preprocessor,
    "agent"
)

customer_type_options = get_categorical_options(
    preprocessor,
    "customer_type"
)


# ============================================================
# 6. BOOKING INFORMATION
# ============================================================

st.subheader("📋 Booking Information")

col1, col2, col3 = st.columns(3)


with col1:

    hotel = st.selectbox(
        "Hotel",
        hotel_options
    )

    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=0,
        max_value=737,
        value=30
    )

    arrival_date_year = st.selectbox(
        "Arrival Year",
        [2015, 2016, 2017],
        index=1
    )

    arrival_date_month = st.selectbox(
        "Arrival Month",
        month_options
    )

    arrival_date_week_number = st.number_input(
        "Arrival Week Number",
        min_value=1,
        max_value=53,
        value=27
    )

    arrival_date_day_of_month = st.number_input(
        "Arrival Day",
        min_value=1,
        max_value=31,
        value=15
    )


with col2:

    stays_in_weekend_nights = st.number_input(
        "Weekend Nights",
        min_value=0,
        max_value=19,
        value=1
    )

    stays_in_week_nights = st.number_input(
        "Week Nights",
        min_value=0,
        max_value=50,
        value=2
    )

    adults = st.number_input(
        "Adults",
        min_value=0,
        max_value=10,
        value=2
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

    babies = st.number_input(
        "Babies",
        min_value=0,
        max_value=10,
        value=0
    )

    meal = st.selectbox(
        "Meal",
        meal_options
    )


with col3:

    country = st.selectbox(
        "Country",
        country_options
    )

    market_segment = st.selectbox(
        "Market Segment",
        market_segment_options
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        distribution_channel_options
    )

    reserved_room_type = st.selectbox(
        "Reserved Room Type",
        room_options
    )

    deposit_type = st.selectbox(
        "Deposit Type",
        deposit_options
    )

    customer_type = st.selectbox(
        "Customer Type",
        customer_type_options
    )


# ============================================================
# 7. BOOKING HISTORY
# ============================================================

st.divider()

st.subheader("📊 Booking History & Behaviour")

col1, col2, col3 = st.columns(3)


with col1:

    is_repeated_guest = st.selectbox(
        "Is Repeated Guest?",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        max_value=30,
        value=0
    )

    previous_bookings_not_canceled = st.number_input(
        "Previous Successful Bookings",
        min_value=0,
        max_value=100,
        value=0
    )


with col2:

    booking_changes = st.number_input(
        "Booking Changes",
        min_value=0,
        max_value=20,
        value=0
    )

    days_in_waiting_list = st.number_input(
        "Days in Waiting List",
        min_value=0,
        max_value=391,
        value=0
    )

    agent = st.selectbox(
        "Booking Agent",
        agent_options
    )


with col3:

    required_car_parking_spaces = st.number_input(
        "Required Parking Spaces",
        min_value=0,
        max_value=8,
        value=0
    )

    total_of_special_requests = st.number_input(
        "Special Requests",
        min_value=0,
        max_value=5,
        value=0
    )

    company_missing = st.selectbox(
        "Company Information",
        [1, 0],
        format_func=lambda x:
            "Missing" if x == 1 else "Available"
    )


# ============================================================
# 8. PRICE INFORMATION
# ============================================================

st.divider()

st.subheader("💰 Price Information")

adr = st.number_input(
    "ADR (Average Daily Rate)",
    min_value=-10.0,
    max_value=5400.0,
    value=100.0,
    step=1.0
)


# ============================================================
# 9. PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Cancellation Risk",
    use_container_width=True
)


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Create raw + engineered features
        # ----------------------------------------------------

        input_df = create_features(

            hotel=hotel,

            lead_time=lead_time,

            arrival_date_year=arrival_date_year,

            arrival_date_month=arrival_date_month,

            arrival_date_week_number=
                arrival_date_week_number,

            arrival_date_day_of_month=
                arrival_date_day_of_month,

            stays_in_weekend_nights=
                stays_in_weekend_nights,

            stays_in_week_nights=
                stays_in_week_nights,

            adults=adults,

            children=children,

            babies=babies,

            meal=meal,

            country=country,

            market_segment=market_segment,

            distribution_channel=
                distribution_channel,

            is_repeated_guest=
                is_repeated_guest,

            previous_cancellations=
                previous_cancellations,

            previous_bookings_not_canceled=
                previous_bookings_not_canceled,

            reserved_room_type=
                reserved_room_type,

            booking_changes=
                booking_changes,

            deposit_type=
                deposit_type,

            agent=agent,

            days_in_waiting_list=
                days_in_waiting_list,

            customer_type=
                customer_type,

            adr=adr,

            required_car_parking_spaces=
                required_car_parking_spaces,

            total_of_special_requests=
                total_of_special_requests,

            company_missing=
                company_missing
        )


        # ----------------------------------------------------
        # Apply trained preprocessing
        # ----------------------------------------------------

        processed_input = preprocessor.transform(
            input_df
        )


        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        cancellation_probability = (
            model.predict_proba(
                processed_input
            )[0][1]
        )


        # ----------------------------------------------------
        # Prediction using optimized threshold = 0.50
        # ----------------------------------------------------

        prediction = int(
            cancellation_probability >= 0.50
        )


        probability_percentage = (
            cancellation_probability * 100
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader("🎯 Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ HIGH CANCELLATION RISK"
            )

            st.metric(
                "Cancellation Probability",
                f"{probability_percentage:.2f}%"
            )

            st.write(
                """
                The model predicts that this booking
                is likely to be cancelled.
                """
            )


        else:

            st.success(
                "✅ LOW CANCELLATION RISK"
            )

            st.metric(
                "Cancellation Probability",
                f"{probability_percentage:.2f}%"
            )

            st.write(
                """
                The model predicts that this booking
                is unlikely to be cancelled.
                """
            )


        # ----------------------------------------------------
        # Risk Level
        # ----------------------------------------------------

        if cancellation_probability >= 0.70:

            risk_level = "High"

        elif cancellation_probability >= 0.40:

            risk_level = "Medium"

        else:

            risk_level = "Low"


        st.info(
            f"Risk Level: **{risk_level}**"
        )


        # ----------------------------------------------------
        # Business Recommendation
        # ----------------------------------------------------

        st.subheader("💡 Business Recommendation")


        if risk_level == "High":

            st.warning(
                """
                Consider additional confirmation measures,
                stricter cancellation policies, or proactive
                customer follow-up for this booking.
                """
            )

        elif risk_level == "Medium":

            st.info(
                """
                Monitor this booking and consider sending
                a confirmation reminder before the arrival date.
                """
            )

        else:

            st.success(
                """
                No immediate additional action is required.
                The booking shows relatively low predicted
                cancellation risk.
                """
            )


        # ----------------------------------------------------
        # Show calculated features
        # ----------------------------------------------------

        with st.expander(
            "View Calculated Features"
        ):

            calculated_features = {

                "Total Guests":
                    input_df["total_guests"].iloc[0],

                "Total Stay Nights":
                    input_df["total_stay_nights"].iloc[0],

                "Lead Time Group":
                    input_df["lead_time_group"].iloc[0],

                "Total Previous Bookings":
                    input_df[
                        "total_previous_bookings"
                    ].iloc[0],

                "Previous Cancellation Rate":
                    input_df[
                        "previous_cancellation_rate"
                    ].iloc[0],

                "ADR per Guest":
                    input_df[
                        "adr_per_guest"
                    ].iloc[0],

                "Arrival Season":
                    input_df[
                        "arrival_season"
                    ].iloc[0]
            }

            st.dataframe(
                pd.DataFrame(
                    calculated_features.items(),
                    columns=[
                        "Feature",
                        "Value"
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "Hotel Booking Cancellation Prediction | "
    "Tuned Random Forest | "
    "F1-Score: 0.8489 | ROC-AUC: 0.9548"
)
