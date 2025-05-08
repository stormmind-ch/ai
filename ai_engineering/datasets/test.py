
from datetime import date, timedelta

def get_past_week_dates(base_date, timespan):
    dates =  [base_date - timedelta(weeks=i) for i in range(timespan + 1)]
    return dates



def get_past_week_dates_year(base_date: date, timespan: int):
    # Step 1: Get the current year's Sundays (backward)
    current_dates = [base_date - timedelta(weeks=i) for i in range(timespan + 1)]

    # Step 2: Get previous year's corresponding Sundays + 1 week
    previous_year_dates = []
    for d in current_dates:
        try:
            one_year_earlier = d.replace(year=d.year - 1)
        except ValueError:
            # handle Feb 29 → Feb 28
            one_year_earlier = d - timedelta(days=365)
        shifted = one_year_earlier + timedelta(weeks=1)
        # Ensure it’s still a Sunday
        if shifted.weekday() != 6:  # Sunday = 6
            # Adjust to next Sunday
            shifted += timedelta(days=(6 - shifted.weekday()))
        previous_year_dates.append(shifted)

    return current_dates + previous_year_dates



# Example usage
base = date(2025, 5, 11)
data = get_past_week_dates_year(base, timespan=2)

print(data)





