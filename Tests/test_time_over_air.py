from pytz import timezone
from datetime import datetime
    
def get_Time_Over_Air(timezone_locale="Turkey", daytime_format="%Y-%m-%d_%H-%M-%S"):
    # "Turkey/Istanbul"
    # "America/New_York"
    datetime_Turkey_Istanbul = datetime.now(
        timezone(timezone_locale)
    )
    return datetime_Turkey_Istanbul.strftime(daytime_format)


print("Time Over Air for Turkey: ", 
    get_Time_Over_Air(
        timezone_locale="Turkey", 
        daytime_format="%Y-%m-%d_%H-%M-%S"
    )
)
