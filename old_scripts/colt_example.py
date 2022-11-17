import calendar
from datetime import date

from colt import from_commandline, Colt


@from_commandline("""
# This is the year you want to print
year = :: int, optional :: >0
""")
def cal(year):
    # which day to start: 0 Monday
    #                     1 Tuesday
    cal = calendar.TextCalendar(6)
    today = date.today()
    if year is not None:
        cal.pryear(today.year)
    else:
        cal.prmonth(today.year, today.month)


class  Cal(Colt):

    _user_input = """
    day =   :: int  :: >-1
    month = -1 :: int :: >-1
    year = -1 :: int :: >-1
    """

    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year

    def __str__(self):
        return f"Cal(day={self.day}, month={self.month}, year={self.year})" 

    @classmethod
    def from_config(cls, config):
        return cls(config['day'], config['month'], config['year'])

if __name__ == '__main__':
    # cal = Cal.from_questions(config='example.ini')
    # print(cal)
    # print("----------")
    cal = Cal.from_commandline()
    print(cal)
