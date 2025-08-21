import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# Dashboard configuration
COLORS = {
    'primary': '#2E5090',
    'secondary': '#0078D4',
    'success': '#107C10',
    'warning': '#FF8C00',
    'danger': '#D13438',
    'info': '#00BCF2',
    'light': '#FAFAFA',
    'dark': '#323130',
    'accent': '#6264A7'
}

# Set page config
st.set_page_config(
    page_title="Roam Policy and Claims Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #323130;
        font-weight: 700;
        margin: 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .radio-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def process_uploaded_data(uploaded_file):
    """Process uploaded CSV and create all necessary columns and flags"""
    try:
        # Read CSV with date parsing
        date_columns = [
            'bookedDateUtc', 'bookedDateLocal', 'arrivalDateUtc', 'arrivalDateLocal',
            'departureDateUtc', 'departureDateLocal', 'cancelledDateUtc', 'cancelledDateLocal',
            'firstPaymentDateLocal', 'pmsDateCreatedUtc', 'pmsDateUpdatedUtc',
            'policyEffectiveDateUtc', 'policyEffectiveDateLocal', 'policyExpirationDateUtc',
            'policyExpirationDateLocal'
        ]
        
        obb = pd.read_csv(uploaded_file, encoding='latin-1', parse_dates=date_columns)
        
        # Create new columns
        obb['bookdays_ahead_arrival'] = (obb['arrivalDateLocal'] - obb['bookedDateLocal']).dt.days
        obb['cancel_before_arrival'] = (obb['arrivalDateLocal'] - obb['cancelledDateLocal']).dt.days
        obb['canceldays_after_booking'] = (obb['cancelledDateLocal'] - obb['bookedDateLocal']).dt.days
        obb['cancelhours_after_booking'] = (obb['cancelledDateLocal'] - obb['bookedDateLocal']).dt.total_seconds() / 3600
        obb['premiums'] = obb['rentTotalWithMarkup'] * 0.057
        
        # Create filter flags
        obb['source_filter'] = obb['SOURCE'].isin(['Vrbo - HomeAway', 'HomeToGo']).astype(int)
        obb['PMmerchant_filter'] = obb['isPMMerchant'].astype(int)
        obb['status_filter'] = (~obb['status'].astype(str).isin(['Hold'])).astype(int)
        obb['48hourcancel_filter'] = ((obb['cancelledDateLocal'].notna()) & (obb['cancelhours_after_booking'] <= 48)).astype(int)
        obb['payment_filter'] = (obb['firstPaymentAmount'] > 0).astype(int)
        obb['window_filter'] = (obb['bookdays_ahead_arrival'] >= 9).astype(int)
        
        # Create month columns
        obb['booking_month'] = obb['bookedDateLocal'].dt.to_period('M').astype(str)
        obb['paid_month'] = obb['firstPaymentDateLocal'].dt.to_period('M').astype(str)
        
        # Filter for paid bookings
        obb_paid = obb[obb['firstPaymentAmount'] > 0].copy()
        
        # Run rebooking analysis
        run_rebooking_analysis(obb_paid)
        
        # Create claims analysis columns
        create_claims_analysis(obb_paid)
        
        return obb_paid
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def run_rebooking_analysis(data):
    """Run rebooking analysis on the data"""
    # Initialize rebooking columns
    data['rebooking_flag'] = 0
    data['rebooked_nights'] = 0
    data['total_rebookings'] = 0
    data['rebooking_details'] = ''
    
    # Get cancelled bookings
    cancelled_mask = data['cancelledDateLocal'].notna()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    cancelled_bookings = data[cancelled_mask]
    total_cancelled = len(cancelled_bookings)
    
    # Process cancelled bookings
    for i, (idx, cancelled_booking) in enumerate(cancelled_bookings.iterrows()):
        if i % 10 == 0:  # Update progress every 10 iterations
            progress_bar.progress((i + 1) / total_cancelled)
            status_text.text(f"Processing rebooking analysis: {i + 1}/{total_cancelled}")
        
        pms_unit_id = cancelled_booking['pmsUnitId']
        cancelled_reservation_id = cancelled_booking['pmsReservationId']
        original_arrival = cancelled_booking['arrivalDateLocal']
        original_departure = cancelled_booking['departureDateLocal']
        cancellation_date = cancelled_booking['cancelledDateLocal']
        
        # Find potential rebookings
        potential_rebookings = data[
            (data['pmsUnitId'] == pms_unit_id) &
            (data['bookedDateLocal'] > cancellation_date) &
            (data['cancelledDateLocal'].isna()) &
            (data['pmsReservationId'] != cancelled_reservation_id)
        ]
        
        if len(potential_rebookings) > 0:
            rebookings_found = 0
            total_rebooked_days = 0
            rebooking_details_list = []
            
            for rebooking_idx, rebooking in potential_rebookings.iterrows():
                rebook_arrival = rebooking['arrivalDateLocal']
                rebook_departure = rebooking['departureDateLocal']
                
                # Check overlap
                overlap_start_date = max(original_arrival.date(), rebook_arrival.date())
                overlap_end_date = min(original_departure.date(), rebook_departure.date())
                
                if overlap_start_date < overlap_end_date:
                    overlapping_days = (overlap_end_date - overlap_start_date).days
                    rebookings_found += 1
                    total_rebooked_days += overlapping_days
                    
                    rebooking_detail = f"RB{rebookings_found}:{rebooking['pmsReservationId']}({overlapping_days}d)"
                    rebooking_details_list.append(rebooking_detail)
            
            # Update records
            if rebookings_found > 0:
                data.loc[idx, 'rebooking_flag'] = 1
                data.loc[idx, 'rebooked_nights'] = total_rebooked_days
                data.loc[idx, 'total_rebookings'] = rebookings_found
                data.loc[idx, 'rebooking_details'] = '; '.join(rebooking_details_list)
    
    progress_bar.progress(1.0)
    status_text.text("Rebooking analysis complete!")

def create_claims_analysis(data):
    """Create claims analysis columns"""
    # Create guest_reasons field
    data['guest_reasons'] = 0
    guest_reason_ids = [41764258, 41877711, 42156340, 42466711]
    data.loc[data['pmsReservationId'].isin(guest_reason_ids), 'guest_reasons'] = 1
    
    # Valid window flag
    data['valid_window_flag'] = (
        (data['cancel_before_arrival'] >= 7) &
        (data['cancel_before_arrival'] <= 60)
    ).astype(int)
    
    # Calculate original nights
    data['original_nights'] = (data['departureDateLocal'].dt.date - data['arrivalDateLocal'].dt.date).apply(lambda x: x.days)
    
    # Full rebooking flag
    data['full_rebooking_flag'] = (
        (data['rebooking_flag'] == 1) &
        (data['rebooked_nights'] == data['original_nights'])
    ).astype(int)
    
    # Eligible for claims
    data['eligible_for_claims'] = (
        (data['valid_window_flag'] == 1) &
        (data['full_rebooking_flag'] == 0) &
        (data['guest_reasons'] == 0)
    ).astype(int)
    
    # Calculate payout components
    data['rent_per_day'] = data['rentTotalWithMarkup'] / data['original_nights']
    data['nights_eligible_payout'] = np.maximum(
        data['original_nights'] - data['rebooked_nights'].fillna(0), 0
    )
    
    # Payment multiplier
    data['payment_multiplier'] = np.where(
        (data['cancel_before_arrival'] >= 7) & (data['cancel_before_arrival'] <= 13),
        0.5,
        np.where(
            (data['cancel_before_arrival'] >= 14) & (data['cancel_before_arrival'] <= 60),
            1.0, 0.0
        )
    )
    
    # Final payout amount
    data['payout_amount'] = np.where(
        data['eligible_for_claims'] == 1,
        data['rent_per_day'] * data['nights_eligible_payout'] * data['payment_multiplier'],
        0
    )

def create_policy_waterfall_plotly(data, selected_month):
    """Create policy waterfall chart using Plotly"""
    if selected_month == 'all':
        filtered_data = data.copy()
        title_suffix = " - All Months"
    else:
        filtered_data = data[data['paid_month'] == selected_month].copy()
        title_suffix = f" - {selected_month}"

    if len(filtered_data) == 0:
        st.warning(f"No data available for {selected_month}")
        return

    # Calculate waterfall values
    total_bookings = len(filtered_data)
    after_source = len(filtered_data[filtered_data['source_filter'] == 1])
    after_pmmerchant = len(filtered_data[(filtered_data['source_filter'] == 1) & (filtered_data['PMmerchant_filter'] == 1)])
    after_status = len(filtered_data[(filtered_data['source_filter'] == 1) & (filtered_data['PMmerchant_filter'] == 1) & (filtered_data['status_filter'] == 1)])
    after_48hourcancel = len(filtered_data[(filtered_data['source_filter'] == 1) & (filtered_data['PMmerchant_filter'] == 1) & (filtered_data['status_filter'] == 1) 
                           & (filtered_data['48hourcancel_filter'] == 0)])
    after_window = len(filtered_data[(filtered_data['source_filter'] == 1) & (filtered_data['PMmerchant_filter'] == 1) & (filtered_data['status_filter'] == 1) 
                         & (filtered_data['48hourcancel_filter'] == 0) & (filtered_data['window_filter'] == 1)])

    # Calculate premiums
    final_filtered = filtered_data[
        (filtered_data['source_filter'] == 1) & 
        (filtered_data['PMmerchant_filter'] == 1) & 
        (filtered_data['status_filter'] == 1) & 
        (filtered_data['48hourcancel_filter'] == 0) & 
        (filtered_data['window_filter'] == 1)]
    total_premiums = final_filtered['premiums'].sum()

    # Create chart data
    categories = ['Total<br>Bookings', 'Source<br>Filter', 'PM Merchant<br>Filter', 'Status<br>Filter', '48hr Cancel<br>Filter', 'Window<br>Filter']
    values = [total_bookings, after_source, after_pmmerchant, after_status, after_48hourcancel, after_window]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['info'], COLORS['warning'], COLORS['danger'], COLORS['success']]

    # Create Plotly bar chart
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, 
               marker_color=colors,
               text=[f'{v:,}' for v in values],
               textposition='outside',
               textfont=dict(size=12, color=COLORS['dark']))
    ])

    fig.update_layout(
        title=f'Policy Waterfall{title_suffix}',
        title_font=dict(size=18, color=COLORS['dark']),
        xaxis_title='',
        yaxis_title='Number of Policies',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500
    )

    fig.update_xaxis(tickfont=dict(size=11, color=COLORS['dark']))
    fig.update_yaxis(tickfont=dict(size=11, color=COLORS['dark']), gridcolor='#E1E1E1')

    st.plotly_chart(fig, use_container_width=True)

    # Show metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Eligible Policies", f"{after_window:,}")
    with col2:
        st.metric("Total Premiums", f"${total_premiums:,.0f}")

def create_claims_waterfall_plotly(data, selected_month):
    """Create claims waterfall chart using Plotly"""
    if selected_month == 'All':
        filtered_data = data.copy()
        title_suffix = " - All Months"
    else:
        filtered_data = data[data['paid_month'] == selected_month].copy()
        title_suffix = f" - {selected_month}"

    # Calculate numbers
    total_cancelled = len(filtered_data[filtered_data['cancelledDateLocal'].notna()])
    valid_window = len(filtered_data[filtered_data['valid_window_flag'] == 1])
    not_rebooked = len(filtered_data[(filtered_data['valid_window_flag'] == 1) & (filtered_data['full_rebooking_flag'] == 0)])
    not_guest_fault = len(filtered_data[(filtered_data['valid_window_flag'] == 1) & (filtered_data['full_rebooking_flag'] == 0) & (filtered_data['guest_reasons'] == 0)])
    eligible = len(filtered_data[filtered_data['eligible_for_claims'] == 1])

    # Create chart data
    categories = ['Total<br>Cancelled', 'Valid<br>Window', 'Not<br>Rebooked', 'Not Guest<br>Fault', 'Eligible<br>Claims']
    values = [total_cancelled, valid_window, not_rebooked, not_guest_fault, eligible]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['info'], COLORS['warning'], COLORS['success']]

    # Create Plotly bar chart
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, 
               marker_color=colors,
               text=[f'{v:,}' if v > 0 else '0' for v in values],
               textposition='outside',
               textfont=dict(size=12, color=COLORS['dark']))
    ])

    fig.update_layout(
        title=f'Claims Waterfall{title_suffix}',
        title_font=dict(size=18, color=COLORS['dark']),
        xaxis_title='',
        yaxis_title='Count',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500
    )

    fig.update_xaxis(tickfont=dict(size=11, color=COLORS['dark']))
    fig.update_yaxis(tickfont=dict(size=11, color=COLORS['dark']), gridcolor='#E1E1E1')

    st.plotly_chart(fig, use_container_width=True)

    return eligible

def show_claims_selector(data, selected_month):
    """Show claims selection interface"""
    if selected_month == 'All':
        return 0, 0
    
    filtered_data = data[data['paid_month'] == selected_month]
    eligible_data = filtered_data[filtered_data['eligible_for_claims'] == 1]
    
    if len(eligible_data) == 0:
        st.warning("No eligible claims found for this month.")
        return 0, 0
    
    st.subheader(f"📋 Select Claims to Pay - {selected_month}")
    
    # Create selection interface
    selected_claims = []
    total_amount = 0
    
    for idx, (_, row) in enumerate(eligible_data.iterrows()):
        col1, col2 = st.columns([1, 9])
        
        with col1:
            selected = st.checkbox("", key=f"claim_{idx}")
        
        with col2:
            claim_info = f"**${row['payout_amount']:.0f}** - ID: {row['pmsReservationId']} | {row['bookedDateLocal'].strftime('%Y-%m-%d')} → {row['cancelledDateLocal'].strftime('%Y-%m-%d')}"
            if pd.notna(row['cancellationNotes']):
                notes = str(row['cancellationNotes'])[:50] + "..." if len(str(row['cancellationNotes'])) > 50 else str(row['cancellationNotes'])
                claim_info += f" | {notes}"
            st.markdown(claim_info)
        
        if selected:
            selected_claims.append(row)
            total_amount += row['payout_amount']
    
    # Show selection summary
    if selected_claims:
        st.success(f"**Selected: {len(selected_claims)} claims | Total: ${total_amount:,.0f}**")
        
        # Show details in expandable section
        with st.expander("View Selected Claims Details"):
            for claim in selected_claims:
                st.write(f"• ID: {claim['pmsReservationId']} - ${claim['payout_amount']:.0f}")
    
    return len(selected_claims), total_amount

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Roam Policy and Claims Analytics</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload CSV File", 
        type=['csv'],
        help="Upload your booking data CSV file to begin analysis"
    )
    
    if uploaded_file is not None:
        # Process data
        with st.spinner("Processing uploaded data..."):
            obb_paid = process_uploaded_data(uploaded_file)
        
        if obb_paid is not None:
            st.success("✅ Data processed successfully!")
            
            # Controls
            col1, col2, col3 = st.columns([2, 6, 2])
            
            with col1:
                # Month selection
                if 'view_type' not in st.session_state:
                    st.session_state.view_type = 'Policy View'
                
                if st.session_state.view_type == 'Policy View':
                    month_options = ['all'] + sorted(obb_paid['paid_month'].dropna().unique().tolist())
                    default_month = 'all'
                else:
                    month_options = ['All'] + sorted(obb_paid['paid_month'].dropna().unique().tolist())
                    default_month = 'All'
                
                selected_month = st.selectbox("📅 Month:", month_options, index=0)
            
            with col2:
                # View selection (centered)
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                view_type = st.radio(
                    "",
                    ['Policy View', 'Claims View'],
                    horizontal=True,
                    key='view_selector'
                )
                st.markdown("</div>", unsafe_allow_html=True)
                st.session_state.view_type = view_type
            
            # Display appropriate view
            if view_type == 'Policy View':
                create_policy_waterfall_plotly(obb_paid, selected_month)
                
            else:  # Claims View
                eligible_count = create_claims_waterfall_plotly(obb_paid, selected_month)
                
                if selected_month != 'All' and eligible_count > 0:
                    st.markdown("---")
                    selected_count, selected_amount = show_claims_selector(obb_paid, selected_month)
                    
                    # Update metrics based on selection
                    if selected_count > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Selected Claims", selected_count)
                        with col2:
                            st.metric("Selected Amount", f"${selected_amount:,.0f}")
    
    else:
        st.info("👆 Please upload a CSV file to begin your analysis")

if __name__ == "__main__":
    main()