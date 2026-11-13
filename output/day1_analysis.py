import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Hotel Analytics Project Started")

df = pd.read_csv("data/hotel_bookings.csv")
inventory = pd.read_csv("data/room_inventory.csv")
print(df.head())
print(inventory.head())

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.info()
print(df.describe())
print(df.isnull().sum())

df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df["agent"] = df["agent"].fillna(0)
df["company"] = df["company"].fillna(0)
print("Missing values handled")

print("Duplicate Records:", df.duplicated().sum())
df = df.drop_duplicates()
print("After Removing Duplicates:", df.shape)

df["arrival_date"] = pd.to_datetime(
    df["arrival_date_year"].astype(str) + "-" +
    df["arrival_date_month"] + "-" +
    df["arrival_date_day_of_month"].astype(str),
    format="%Y-%B-%d",
    errors="coerce"
)

df["total_nights"] = (
    df["stays_in_weekend_nights"] +
    df["stays_in_week_nights"]
)

df["departure_date"] = (
    df["arrival_date"] +
    pd.to_timedelta(df["total_nights"], unit="D")
)

print(df[["arrival_date", "departure_date"]].head())

df["booking_state"] = np.select(
    [
        df["is_canceled"] == 1,
        df["reservation_status"] == "No-Show"
    ],
    [
        "Cancelled",
        "Cancelled"
    ],
    default="Booked"
)

print(df["booking_state"].value_counts())

print(df.columns.tolist())


valid = df[
    (df["is_canceled"] == 0) &
    (df["reservation_status"] != "No-Show") &
    (df["total_nights"] > 0)
].copy()

# Sort reservations

valid = valid.sort_values(
    ["reserved_room_type", "arrival_date"]
)

# Find possible date overlaps in same room category

valid["previous_max_departure"] = (
    valid.groupby("reserved_room_type")["departure_date"]
    .transform(lambda x: x.cummax().shift())
)

valid["possible_overlap"] = (
    valid["arrival_date"] <
    valid["previous_max_departure"]
)

print(
    "Possible category-level overlaps:",
    valid["possible_overlap"].sum()
)

total_bookings = len(df)

print("Total Bookings:", total_bookings)

cancelled_bookings = (
    df["is_canceled"] == 1
).sum()

print("Cancelled Bookings:", cancelled_bookings)

non_cancelled = (
    df["is_canceled"] == 0
).sum()

print("Non-Cancelled Bookings:", non_cancelled)

cancellation_rate = (
    cancelled_bookings / total_bookings
) * 100

print("Cancellation Rate:", round(cancellation_rate, 2), "%")

total_room_nights = df.loc[
    df["is_canceled"] == 0,
    "total_nights"
].sum()

print("Total Room Nights:", total_room_nights)

room_summary = df.groupby(
    "reserved_room_type"
).agg(
    total_bookings=("hotel", "count"),
    cancelled_bookings=("is_canceled", "sum"),
    average_rate=("adr", "mean")
).reset_index()

room_summary["cancellation_rate"] = (
    room_summary["cancelled_bookings"] /
    room_summary["total_bookings"] * 100
)

print(room_summary)

# Keep valid stays

stays = df[
    (df["is_canceled"] == 0) &
    (df["reservation_status"] != "No-Show") &
    (df["total_nights"] > 0)
].copy()

# Create one row for each occupied date

daily_records = []

for _, row in stays.iterrows():

    dates = pd.date_range(
        start=row["arrival_date"],
        end=row["departure_date"] - pd.Timedelta(days=1)
    )

    for date in dates:
        daily_records.append({
            "date": date,
            "room_category": row["reserved_room_type"],
            "hotel": row["hotel"]
        })

daily_occupancy = pd.DataFrame(daily_records)

# Count occupied reservations by date and category

daily_summary = daily_occupancy.groupby(
    ["date", "room_category"]
).size().reset_index(name="occupied_booking_count")

print(daily_summary.head(10))

# Create output folder

os.makedirs("output", exist_ok=True)

# Save cleaned dataset

df.to_csv(
    "output/cleaned_hotel_bookings.csv",
    index=False
)

# Save room summary

room_summary.to_csv(
    "output/room_category_summary.csv",
    index=False
)

# Save daily occupancy records

daily_summary.to_csv(
    "output/daily_occupancy_summary.csv",
    index=False
)

print("All output files saved successfully!")