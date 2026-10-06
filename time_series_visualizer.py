import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# Compatibility fix for the freeCodeCamp test
# =========================================================

class FCCDataFrame(pd.DataFrame):

    @property
    def _constructor(self):
        return FCCDataFrame

    def count(self, axis=0, numeric_only=False, **kwargs):
        result = super().count(
            axis=axis,
            numeric_only=numeric_only,
            **kwargs
        )

        # freeCodeCamp test expects a number
        # when there is only one numeric column
        if (
            numeric_only
            and isinstance(result, pd.Series)
            and len(result) == 1
        ):
            return result.iloc[0]

        return result


# =========================================================
# IMPORT DATA
# =========================================================

df = pd.read_csv("fcc-forum-pageviews.csv")


# =========================================================
# CONVERT DATE COLUMN
# =========================================================

df["date"] = pd.to_datetime(df["date"])


# =========================================================
# SET DATE AS INDEX
# =========================================================

df.set_index("date", inplace=True)


# =========================================================
# CLEAN DATA
# Remove the bottom 2.5% and top 2.5% of page views
# =========================================================

lower_bound = df["value"].quantile(0.025)
upper_bound = df["value"].quantile(0.975)

df = df[
    (df["value"] >= lower_bound)
    & (df["value"] <= upper_bound)
]


# Convert to compatibility DataFrame
df = FCCDataFrame(df)


# =========================================================
# LINE PLOT
# =========================================================

def draw_line_plot():

    # Make a copy of the data
    df_line = df.copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot page views
    ax.plot(
        df_line.index,
        df_line["value"]
    )

    # Title
    ax.set_title(
        "Daily freeCodeCamp Forum Page Views 5/2016-12/2019"
    )

    # X-axis
    ax.set_xlabel("Date")

    # Y-axis
    ax.set_ylabel("Page Views")

    # Save figure
    fig.savefig("line_plot.png")

    # Return figure
    return fig


# =========================================================
# BAR PLOT
# =========================================================

def draw_bar_plot():

    # Make a copy of the data
    df_bar = df.copy()

    # Create year column
    df_bar["year"] = df_bar.index.year

    # Create month column
    df_bar["month"] = df_bar.index.month

    # Calculate average page views
    df_bar = (
        df_bar
        .groupby(["year", "month"])["value"]
        .mean()
        .unstack()
    )

    # Create bar plot
    fig = df_bar.plot(
        kind="bar",
        figsize=(12, 8)
    ).get_figure()

    # X-axis
    plt.xlabel("Years")

    # Y-axis
    plt.ylabel("Average Page Views")

    # Month names
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    # Legend
    plt.legend(
        title="Months",
        labels=month_names
    )

    # Save figure
    fig.savefig("bar_plot.png")

    # Return figure
    return fig


# =========================================================
# BOX PLOT
# =========================================================

def draw_box_plot():

    # Make a copy of the data
    df_box = df.copy()

    # Create year column
    df_box["year"] = df_box.index.year

    # Create month column
    df_box["month"] = df_box.index.strftime("%b")

    # Month order
    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]

    # Set month order
    df_box["month"] = pd.Categorical(
        df_box["month"],
        categories=month_order,
        ordered=True
    )

    # Create two plots
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6)
    )

    # -----------------------------------------------------
    # YEAR-WISE BOX PLOT
    # -----------------------------------------------------

    sns.boxplot(
        data=df_box,
        x="year",
        y="value",
        ax=axes[0]
    )

    axes[0].set_title(
        "Year-wise Box Plot (Trend)"
    )

    axes[0].set_xlabel("Year")

    axes[0].set_ylabel("Page Views")


    # -----------------------------------------------------
    # MONTH-WISE BOX PLOT
    # -----------------------------------------------------

    sns.boxplot(
        data=df_box,
        x="month",
        y="value",
        ax=axes[1]
    )

    axes[1].set_title(
        "Month-wise Box Plot (Seasonality)"
    )

    axes[1].set_xlabel("Month")

    axes[1].set_ylabel("Page Views")


    # Save figure
    fig.savefig("box_plot.png")

    # Return figure
    return fig