import datetime


def format_time(second_elappsed):
    return datetime.timedelta(seconds=int(round(second_elappsed)))
