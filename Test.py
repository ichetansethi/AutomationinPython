# Use this file to locally test the Streamlit dashboard before deploying it to Streamlit Sharing.
# Use this file to debug and make changes to the dashboard code.

from io import BytesIO
from typing import Optional, Dict, List, Any, Tuple
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from datetime import date
from plotly.graph_objs._figure import Figure

class BugTracker:
    """
    A Streamlit dashboard application for analyzing and visualizing issue tracking data.
    
    This class provides functionality to:
    - Load and validate issue tracking data from CSV/Excel files
    - Filter data by date range and categories
    - Create various visualizations including time series, treemaps, and pie charts
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
        "Bug Type",
        "Mode of Reporting",
        "Resolution Time (Mins)"
    ]
    
    PAGE_TITLE = 'Issue Summary Dashboard'
    PAGE_ICON = ':bar_chart:'
    FILTER_COLUMNS = ["Reporting Category", "Bug Type", "Mode of Reporting"]

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
        Create a time series chart showing resolution times over time.
        
        Args:
            df: DataFrame containing the filtered data
            
        Returns:
            Figure: Plotly figure object containing the time series chart
        """
        df = df.copy()
        df["month_year"] = df["Date of Reporting"].dt.to_period("M")
        linechart = pd.DataFrame(
            df.groupby(df["month_year"].dt.strftime("%Y : %b"))["Resolution Time (Mins)"]
            .sum()
        ).reset_index()
        
        return px.line(
            linechart,
            x="month_year",
            y="Resolution Time (Mins)",
            labels={"Resolution Time (Mins)": "Total Resolution Time"},
            height=500,
            template="gridon"
        )

    def create_treemap(self, df: pd.DataFrame) -> Optional[Figure]:
        """
        Create a treemap visualization of resolution time hierarchy.
        
        Args:
            df: DataFrame containing the filtered data
            
        Returns:
            Optional[Figure]: Plotly figure object containing the treemap or None if creation fails
        """
        try:
            treemap = px.treemap(
                df,
                path=["Mode of Reporting", "Reporting Category", "Bug Type"],
                values="Resolution Time (Mins)",
                color="Bug Type"
            )
            treemap.update_layout(height=650)
            return treemap
        except Exception as e:
            st.error(f"Error creating treemap: {str(e)}")
            return None

    def create_pie_charts(self, df: pd.DataFrame) -> Tuple[Figure, Figure]:
        """
        Create pie charts for resolution time distribution.
        
        Args:
            df: DataFrame containing the filtered data
            
        Returns:
            Tuple[Figure, Figure]: Category pie chart and bug type pie chart
        """
        category_pie = px.pie(
            df,
            values="Resolution Time (Mins)",
            names="Reporting Category",
            template="gridon"
        )
        
        bug_pie = px.pie(
            df,
            values="Resolution Time (Mins)",
            names="Bug Type",
            template="plotly_dark"
        )
        
        return category_pie, bug_pie

    def create_summary_tables(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create summary tables for issues by category and bug type.
        
        Args:
            df: DataFrame containing the filtered data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Category summary and bug type summary
        """
        category_summary = df.groupby("Reporting Category")["Resolution Time (Mins)"].agg(['count', 'mean'])
        bug_summary = df.groupby("Bug Type")["Resolution Time (Mins)"].agg(['count', 'mean'])
        return category_summary, bug_summary

    def setup_date_filters(self, df: pd.DataFrame) -> Tuple[date, date]:
        """
        Set up date range filters for the dashboard.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Tuple[date, date]: Selected start and end dates
        """
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                df["Date of Reporting"].min().date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                df["Date of Reporting"].max().date()
            )
            
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
        
        # Time Series Analysis
        st.subheader('Time Series Analysis')
        st.plotly_chart(self.create_time_series_chart(df), use_container_width=True)
        
        # Treemap
        st.subheader("Resolution Time Hierarchy")
        treemap = self.create_treemap(df)
        if treemap:
            st.plotly_chart(treemap, use_container_width=True)
        
        # Pie Charts
        col1, col2 = st.columns(2)
        category_pie, bug_pie = self.create_pie_charts(df)
        
        with col1:
            st.subheader('Resolution Time by Reporting Category')
            st.plotly_chart(category_pie, use_container_width=True)
        
        with col2:
            st.subheader('Resolution Time by Bug Type')
            st.plotly_chart(bug_pie, use_container_width=True)
        
        # Summary Tables
        col1, col2 = st.columns(2)
        category_summary, bug_summary = self.create_summary_tables(df)
        
        with col1:
            st.subheader('Issues by Category')
            st.dataframe(category_summary.style.background_gradient(cmap="Blues"))
        
        with col2:
            st.subheader('Issues by Bug Type')
            st.dataframe(bug_summary.style.background_gradient(cmap="Oranges"))
        
        # Export Options
        st.subheader("Data Export")
        self.export_data(df)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dashboard = BugTracker()
    dashboard.run()