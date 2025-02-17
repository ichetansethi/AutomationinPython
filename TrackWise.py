from io import BytesIO
from typing import Optional, Dict, List, Any, Tuple
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import date
from plotly.graph_objs._figure import Figure

class BugTracker:
    """
    A Streamlit dashboard application for analyzing and visualizing issue tracking data.
    
    This class provides functionality to:
    - Load and validate issue tracking data from CSV/Excel files
    - Filter data by date range and categories
    - Create various visualizations including time series, bar graphs, line charts and pie charts
    - Export filtered data in multiple formats
    
    Attributes:
        REQUIRED_COLUMNS (List[str]): List of required columns in the input data
        PAGE_TITLE (str): Title of the Streamlit dashboard
        PAGE_ICON (str): Icon for the dashboard
        FILTER_COLUMNS (List[str]): Columns available for filtering
    """
    
    REQUIRED_COLUMNS = [
        "Date of Reporting",
        "Reporting Category",
        "Bug Severity",
        "Mode of Reporting",
        "Resolution Time (Mins)"
    ]
    
    PAGE_TITLE = 'Issue Summary Dashboard'
    PAGE_ICON = ':bar_chart:'
    FILTER_COLUMNS = ["Reporting Category", "Bug Severity", "Mode of Reporting"]

    def __init__(self):
        """Initialize the dashboard with basic Streamlit configuration."""
        self.df: Optional[pd.DataFrame] = None
        self._configure_page()

    def _configure_page(self) -> None:
        """Configure the Streamlit page layout and styling."""
        st.set_page_config(
            page_title=self.PAGE_TITLE,
            page_icon=self.PAGE_ICON,
            layout='wide'
        )
        st.title(f"{self.PAGE_ICON} {self.PAGE_TITLE}")
        
        # Apply custom CSS
        st.markdown(
            '<style>div.block-container{padding-top:2em; text-align: center;} h1 {margin-top: 0;}</style>',
            unsafe_allow_html=True,
        )

    @staticmethod
    @st.cache_data
    def load_data(file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[pd.DataFrame]:
        """
        Load and validate data from an uploaded file.
        
        Args:
            file: Uploaded file object (CSV or Excel)
            
        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None if loading fails
        """
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format")
                
            # Validate required columns
            missing_columns = [col for col in BugTracker.REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return None

            # Process dates
            df["Date of Reporting"] = pd.to_datetime(df["Date of Reporting"], errors="coerce")
            invalid_dates = df["Date of Reporting"].isna().sum()
            if invalid_dates > 0:
                st.warning(f"{invalid_dates} invalid date(s) found in 'Date of Reporting'. These entries will be excluded.")

            return df

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def create_time_series_chart(self, df: pd.DataFrame) -> Figure:
        """
        Create an enhanced time series chart showing average resolution time trends,
        including issue count and rolling average.

        Args:
            df: DataFrame containing the filtered data

        Returns:
            Figure: Plotly figure object containing the time series chart
        """
        df = df.copy()
        df["month_year"] = df["Date of Reporting"].dt.to_period("M")

        # Group by month
        linechart = (
            df.groupby("month_year")
            .agg({"Resolution Time (Mins)": "mean", "Bug Severity": "count"})  # Average resolution time & issue count
            .reset_index()
            .sort_values("month_year")
        )

        # Rolling average for trend smoothing
        linechart["Rolling Avg"] = linechart["Resolution Time (Mins)"].rolling(window=3, min_periods=1).mean()

        # Convert to string for display
        linechart["month_year_str"] = linechart["month_year"].astype(str)

        # Create figure
        fig = go.Figure()

        # Add average resolution time line
        fig.add_trace(go.Scatter(
            x=linechart["month_year_str"], 
            y=linechart["Resolution Time (Mins)"], 
            mode="lines+markers", 
            name="Avg Resolution Time",
            line=dict(color="blue")
        ))

        # Add rolling average line
        fig.add_trace(go.Scatter(
            x=linechart["month_year_str"], 
            y=linechart["Rolling Avg"], 
            mode="lines", 
            name="3-Month Rolling Avg",
            line=dict(color="red", dash="dash")
        ))

        # Add issue count as bar chart
        fig.add_trace(go.Bar(
            x=linechart["month_year_str"], 
            y=linechart["Bug Severity"], 
            name="Number of Issues",
            marker_color="green",
            opacity=0.6,
            yaxis="y2"
        ))

        # Update layout for dual-axis
        fig.update_layout(
            title="Time Series Analysis: Avg Resolution Time vs. Issue Count",
            xaxis_title="Month-Year",
            yaxis=dict(title="Avg Resolution Time (Mins)", side="left"),
            yaxis2=dict(title="Number of Issues", overlaying="y", side="right"),
            height=500,
            template="gridon"
        )

        return fig

    def create_issue_distribution(self, df: pd.DataFrame) -> Optional[Figure]:
        """
        Create a grouped bar chart showing the distribution of issues across reporting categories
        and modes of reporting.

        Args:
            df: DataFrame containing the filtered data
                Expected columns: 'Mode of Reporting', 'Reporting Category'
                
        Returns:
            Optional[Figure]: Plotly figure object containing the bar chart or None if creation fails
        """
        try:
            # Group the data to get counts for each combination
            grouped_data = df.groupby(['Reporting Category', 'Mode of Reporting']).size().reset_index(name='Count')
            
            # Create a grouped bar chart
            fig = px.bar(
                grouped_data,
                x='Reporting Category',
                y='Count',
                color='Mode of Reporting',
                title='Issue Distribution by Category and Reporting Mode',
                labels={
                    'Reporting Category': 'Category of Issue',
                    'Count': 'Number of Issues Reported',
                    'Mode of Reporting': 'Reporting Mode'
                },
                height=650
            )
            
            # Enhance the layout for better readability
            fig.update_layout(
                barmode='group',  # Group bars for each category
                xaxis_tickangle=-45,  # Angle the x-axis labels for better readability
                legend_title_text='Mode of Reporting',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return fig  # Return the figure if successful
            
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None  # Return None if there's an error
        
    def prepare_severity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for the dual-axis visualization of bug severity vs count and resolution time.
        
        This function processes the input DataFrame to calculate both the count of bugs and
        their mean resolution time for each severity level (P1-P4).
        
        Args:
            df: DataFrame containing the bug data
                Expected columns: 'Bug Severity', 'Resolution Time (Mins)'
                
        Returns:
            pd.DataFrame: Processed DataFrame with columns for severity, count, and mean resolution time
        """
        # Filter for P1-P4 severity levels and remove nulls
        severity_df = df[df['Bug Severity'].isin(['P1', 'P2', 'P3', 'P4'])].copy()
        
        # Create summary DataFrame with both metrics
        severity_analysis = severity_df.groupby('Bug Severity').agg({
            'Resolution Time (Mins)': ['count', 'mean']
        }).reset_index()
        
        # Flatten column names and rename for clarity
        severity_analysis.columns = pd.Index(['Bug Severity', 'Count of Bugs', 'Mean Resolution Time'])
        
        # Sort by severity level to ensure correct order
        severity_order = ['P1', 'P2', 'P3', 'P4']
        severity_analysis['Bug Severity'] = pd.Categorical(
            severity_analysis['Bug Severity'],
            categories=severity_order,
            ordered=True
        )
        severity_analysis = severity_analysis.sort_values('Bug Severity')
        
        # Round mean resolution time for better readability
        severity_analysis['Mean Resolution Time'] = severity_analysis['Mean Resolution Time'].round(2)
        
        return severity_analysis

    def create_severity_pie(self, df: pd.DataFrame) -> Optional[Figure]:
        """
        Create a pie chart showing the distribution of bugs by severity level (P1-P4).
        
        Args:
            df: DataFrame containing the filtered data
                Expected to have a column for bug severity
                
        Returns:
            Optional[Figure]: Plotly figure object containing the pie chart or None if creation fails
        """
        try:
            # Filter the DataFrame to include only P1-P4 severity levels and remove null values
            severity_data = df[df['Bug Severity'].isin(['P1', 'P2', 'P3', 'P4'])].copy()
            
            # Group by severity to get the count of bugs in each category
            severity_counts = severity_data['Bug Severity'].value_counts().reset_index()
            severity_counts.columns = pd.Index(['Bug Severity', 'Count'])
            
            # Sort the data to ensure P1-P4 appear in order
            severity_counts['Bug Severity'] = pd.Categorical(
                severity_counts['Bug Severity'], 
                categories=['P1', 'P2', 'P3', 'P4'], 
                ordered=True
            )
            severity_counts = severity_counts.sort_values('Bug Severity')
            
            # Create a color map for severity levels
            color_map = {
                'P1': '#FF4444',  # Red for highest severity
                'P2': '#FFA500',  # Orange for high severity
                'P3': '#FFD700',  # Yellow for medium severity
                'P4': '#90EE90'   # Light green for low severity
            }
            
            # Create the pie chart with custom styling
            fig = px.pie(
                severity_counts,
                values='Count',
                names='Bug Severity',
                title='Bug Distribution by Severity Level',
                color='Bug Severity',
                color_discrete_map=color_map
            )
            
            # Enhance the layout for better readability
            fig.update_layout(
                showlegend=True,
                legend_title_text='Severity Level',
                annotations=[dict(
                    text='P1: Critical<br>P2: High<br>P3: Medium<br>P4: Low',
                    xref='paper',
                    yref='paper',
                    x=1.1,
                    y=0.5,
                    showarrow=False,
                    align='right'
                )],
                height=500,
                width=700
            )
            
            # Add percentage and count in hover text
            fig.update_traces(
                texttemplate='%{label}<br>%{percent:.1%}<br>(%{value:,} issues)',
                hovertemplate='%{label}<br>Count: %{value:,}<br>Percentage: %{percent:.1%}'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating severity pie chart: {str(e)}")
            return None

    def create_summary_tables(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create summary tables for issues by category and bug severity with custom column names.

        Args:
            df: DataFrame containing the filtered data
                Expected columns: 'Reporting Category', 'Bug Severity', 'Resolution Time (Mins)'
                
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Category summary and bug severity summary with renamed columns
        """
        # Create category summary with counts and mean resolution time
        category_summary = df.groupby("Reporting Category").agg({
            'Resolution Time (Mins)': ['size', 'mean']  # Using size instead of count to get total issues
        })
        category_summary.columns = pd.Index(['Count of Issues', 'Mean Resolution Time'])

        # Create bug severity summary with counts and mean resolution time
        bug_summary = df.groupby("Bug Severity").agg({
            'Resolution Time (Mins)': ['size', 'mean']  # Using size instead of count to get total issues
        })
        bug_summary.columns = pd.Index(['Count of Issues', 'Mean Resolution Time'])

        # Round mean resolution times to 2 decimal places for better readability
        category_summary['Mean Resolution Time'] = category_summary['Mean Resolution Time'].round(2)
        bug_summary['Mean Resolution Time'] = bug_summary['Mean Resolution Time'].round(2)

        return category_summary, bug_summary

    def prepare_category_time_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for dual-line visualization comparing average resolution and response times by category.

        This function processes the input DataFrame to calculate mean resolution and response times
        for each category, providing a comprehensive view of time metrics across different issue types.

        Args:
            df: DataFrame containing the issue data
                Expected columns: 'Reporting Category', 'Resolution Time (Mins)', 'Response Time (Mins)'
                
        Returns:
            pd.DataFrame: Processed DataFrame with columns for category and both time metrics
        """
        # Calculate mean times for each category
        category_analysis = df.groupby('Reporting Category').agg({
            'Resolution Time (Mins)': 'mean',
            'Response Time (Mins)': 'mean'
        }).reset_index()

        # Rename columns for clarity
        category_analysis.columns = pd.Index(['Category', 'Avg Resolution Time', 'Avg Response Time'])

        # Round the time values for better readability
        category_analysis['Avg Resolution Time'] = category_analysis['Avg Resolution Time'].round(2)
        category_analysis['Avg Response Time'] = category_analysis['Avg Response Time'].round(2)

        # Sort by average resolution time to highlight categories needing most attention
        category_analysis = category_analysis.sort_values('Avg Resolution Time', ascending=False)

        return category_analysis

    def setup_date_filters(self, df: pd.DataFrame) -> Tuple[date, date]:
        """
        Set up date range filters for the dashboard.
        
        Args:
            df: DataFrame containing the data
                Expected to have 'Date of Reporting' column
                
        Returns:
            Tuple[date, date]: Selected start and end dates
        """
        # Get min and max dates from DataFrame
        min_date = df["Date of Reporting"].min().date()
        max_date = df["Date of Reporting"].max().date()
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date_input = st.date_input(
                "Start Date",
                min_date
            )
            # Ensure we have a date object
            start_date = start_date_input if isinstance(start_date_input, date) else min_date
        
        with col2:
            end_date_input = st.date_input(
                "End Date",
                max_date
            )
            # Ensure we have a date object
            end_date = end_date_input if isinstance(end_date_input, date) else max_date
        
        return start_date, end_date
    
    def setup_sidebar_filters(self, df: pd.DataFrame) -> Dict[str, List[Any]]:
        """
        Set up sidebar filters for the dashboard.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Dict[str, List[Any]]: Dictionary of selected filter values
        """
        st.sidebar.header("Filters")
        filters = {}
        
        for column in self.FILTER_COLUMNS:
            if column in df.columns:
                filters[column] = st.sidebar.multiselect(
                    f"Select {column}",
                    df[column].dropna().unique()
                )
                
        return filters

    def export_data(self, df: pd.DataFrame) -> None:
        """
        Create data export buttons for CSV and Excel formats.
        
        Args:
            df: DataFrame to export
        """
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                'Download Filtered Data (CSV)',
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        
        with col2:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button(
                'Download Filtered Data (Excel)',
                data=output.getvalue(),
                file_name="filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    def run(self) -> None:
        """
        Main method to run the dashboard application.
        """
        uploaded_file = st.file_uploader(":file_folder: Upload your file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is None:
            st.info("Please upload a CSV, XLS, or XLSX file to begin analysis.")
            return
            
        df = self.load_data(uploaded_file)
        
        if df is None or df.empty:
            st.error("Unable to process the uploaded file. Please check the file format and contents.")
            return
            
        # Set up filters
        start_date, end_date = self.setup_date_filters(df)
        df = df[
            (df["Date of Reporting"].dt.date >= start_date) &
            (df["Date of Reporting"].dt.date <= end_date)
        ]
        
        filters = self.setup_sidebar_filters(df)
        for column, selected_values in filters.items():
            if selected_values:
                df = df[df[column].isin(selected_values)]
        
        if df.empty:
            st.warning("No data available after applying filters.")
            return
        
        # Display filtered data and visualizations
        st.write("Filtered Data Preview:")
        st.dataframe(df.head())
        
        # Export Options
        st.subheader("Data Export")
        self.export_data(df)

        # Time Series Analysis
        st.subheader('Time Series Analysis')
        st.plotly_chart(self.create_time_series_chart(df), use_container_width=True)

        # Summary Tables
        col1, col2 = st.columns(2)
        category_summary, bug_summary = self.create_summary_tables(df)

        with col1:
            st.subheader('Issues by Category')
            st.dataframe(category_summary.style.background_gradient
            (cmap='RdYlGn_r',subset=['Mean Resolution Time']).background_gradient
            (cmap='Blues',subset=['Count of Issues']).format({'Count of Issues': '{:,.0f}', 'Mean Resolution Time': '{:,.2f}'}))

        with col2:
            st.subheader('Issues by Bug Severity')
            st.dataframe(bug_summary.style.background_gradient
            (cmap='RdYlGn_r', subset=['Mean Resolution Time']).background_gradient
            (cmap='Blues',subset=['Count of Issues']).format({'Count of Issues': '{:,.0f}','Mean Resolution Time': '{:,.2f}'}))

        # Issue Distribution Visualization
        st.subheader("Issue Distribution by Category and Reporting Mode")
        bar_chart = self.create_issue_distribution(df)
        if bar_chart:
            st.plotly_chart(bar_chart, use_container_width=True)
        st.caption("This graph shows how many issues were reported through different modes across each category. Taller bars indicate more issues reported through that particular mode.")
            
        # Create columns for better visual organization
        col1, col2 = st.columns([2, 1])

        # Create the severity pie chart
        severity_pie = self.create_severity_pie(df)

        with col1:
            st.subheader('Bug Distribution by Severity Level')
        if severity_pie:
            st.plotly_chart(severity_pie, use_container_width=True)

        with col2:
            st.subheader('Severity Level Guide')
        st.markdown("""
            **P1 - Critical**
            - Immediate attention required
            - System-critical bugs
            - Major business impact
            
            **P2 - High**
            - Significant functionality affected
            - High business impact
            - Needs quick resolution
            
            **P3 - Medium**
            - Moderate impact on functionality
            - Regular priority fixes
            - Non-critical features affected
            
            **P4 - Low**
            - Minor issues
            - Minimal business impact
            - Can be addressed in regular maintenance
        """)
        
        # Create the dual-axis visualization for bug severity analysis
        st.subheader("Bug Distribution and Resolution Time by Severity")

        # Get the prepared data
        severity_analysis = self.prepare_severity_analysis(df)

        # Create the figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar chart for bug counts
        fig.add_trace(
        go.Bar(
            x=severity_analysis['Bug Severity'],
            y=severity_analysis['Count of Bugs'],
            name="Count of Bugs",
            marker_color='rgb(158,202,225)',
            text=severity_analysis['Count of Bugs'],
            textposition='auto',
        ),
        secondary_y=False
        )

        # Add line chart for resolution time
        fig.add_trace(
        go.Scatter(
            x=severity_analysis['Bug Severity'],
            y=severity_analysis['Mean Resolution Time'],
            name="Mean Resolution Time",
            line=dict(color='rgb(255,127,14)', width=3),
            mode='lines+markers+text',
            text=severity_analysis['Mean Resolution Time'].round(1),
            textposition='top center',
            marker=dict(size=10)
        ),
        secondary_y=True
        )

        # Update layout with titles and labels
        fig.update_layout(
        title_text="Bug Count and Resolution Time Analysis by Severity",
        showlegend=True,
        height=500,
        bargap=0.3,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
        )

        # Update axes titles
        fig.update_xaxes(title_text="Bug Severity Level")
        fig.update_yaxes(
        title_text="Count of Bugs", 
        secondary_y=False,
        gridcolor='lightgray'
        )
        fig.update_yaxes(
        title_text="Mean Resolution Time (mins)", 
        secondary_y=True,
        gridcolor='lightgray'
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Add explanatory note
        st.caption("""
        This visualization shows the relationship between bug severity and resolution metrics:
        - Blue bars represent the count of bugs for each severity level
        - Orange line shows the mean resolution time in minutes
        - P1 represents the highest severity while P4 represents the lowest
        """)

        # Create the dual-line visualization for time analysis
        st.subheader("Resolution and Response Time Analysis by Category")

        # Get the prepared data
        category_analysis = self.prepare_category_time_analysis(df)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add line for average resolution time
        fig.add_trace(
        go.Scatter(
            x=category_analysis['Category'],
            y=category_analysis['Avg Resolution Time'],
            name="Average Resolution Time",
            line=dict(color='rgb(31, 119, 180)', width=3),  # Blue line
            mode='lines+markers+text',
            text=category_analysis['Avg Resolution Time'].round(1),
            textposition='top center',
            marker=dict(size=8)
        ),
        secondary_y=False
        )

        # Add line for average response time
        fig.add_trace(
        go.Scatter(
            x=category_analysis['Category'],
            y=category_analysis['Avg Response Time'],
            name="Average Response Time",
            line=dict(color='rgb(255, 127, 14)', width=3),  # Orange line
            mode='lines+markers+text',
            text=category_analysis['Avg Response Time'].round(1),
            textposition='bottom center',
            marker=dict(size=8)
        ),
        secondary_y=True
        )

        # Update layout with titles and labels
        fig.update_layout(
        title_text="Average Resolution and Response Times by Category",
        showlegend=True,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        # Add more whitespace on bottom for labels
        margin=dict(b=100)
        )

        # Customize the axes
        fig.update_xaxes(
        title_text="Issue Category",
        tickangle=45  # Angle the category labels for better readability
        )
        fig.update_yaxes(
        title_text="Average Resolution Time (mins)", 
        secondary_y=False,
        gridcolor='lightgray',
        zeroline=True
        )
        fig.update_yaxes(
        title_text="Average Response Time (mins)", 
        secondary_y=True,
        gridcolor='lightgray',
        zeroline=True
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Add explanatory note
        st.caption("""
        This visualization compares two key time metrics across issue categories:
        - Blue line: Average time taken to resolve issues (Resolution Time)
        - Orange line: Average time taken to initially respond to issues (Response Time)
        Higher values indicate areas that might need process improvements.
        """)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dashboard = BugTracker()
    dashboard.run()