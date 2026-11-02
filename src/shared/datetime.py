#create datetime
import pandas as pd

def create_datetime(df, date_col="Date", time_col=None):
    date_part = df[date_col].astype(str)

    if time_col is not None and time_col in df.columns:
        time_part = (
            df[time_col]
            .astype(str)
            .str.strip()
            .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        )

        # Convert times like "930" to "09:30"
        time_part = time_part.where(
            time_part.str.contains(":", na=False),
            time_part.str.zfill(4).str[:2] + ":" + time_part.str[-2:]
        )

        # Default missing times to midnight
        time_part = time_part.fillna("00:00:00")
        

    else:
        # No time column supplied: default to midnight
        time_part = pd.Series("00:00:00", index=df.index)

    df[date_col] = pd.to_datetime(
        date_part + " " + time_part,
        errors="coerce"
    )

    return df