# https://www.deep-ml.com/problems/92

import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	days = list(range(1,11))
    n = len(days)
    detrended = []
    for i, cons in zip(days, consumption_data):
        fluctuation_i = 10*math.sin((2*PI*i)/10)
        detrended_value = cons - fluctuation_i
        detrended.append(detrended_value)

    sum_x = sum(days)
    sum_y = sum(detrended)
    sum_xy = sum(x*y for x, y in zip(days, detrended))
    sum_x2 = sum(x**2 for x in days)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    day_15_base = m*15 + b
    day_15_fluctuation = 10*math.sin((2*PI*15)/10)
    day_15_prediction = day_15_base + day_15_fluctuation

    day_15_rounded = round(day_15_prediction)
    final_15 = math.ceil(day_15_rounded * 1.05)

    return final_15
