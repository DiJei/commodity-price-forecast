from .google_news import GoogleNews
from datetime import date
from datetime import timedelta
import pandas as pd
import logging
import time

# log configuration
logging.basicConfig(level = logging.INFO)

class HeadlineSeeker():
    
    def __init__(self,enable_prox = False, url_pro = []):
        self.googlenews = GoogleNews(enable_prox, url_pro)
        self.keywords = []
        
    def set_keywords(self, keywords_list):
        self.keywords = keywords_list
        
    def set_debug(self, bo):
        if bo:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def get_daily_news(self, lang = 'pt'):
        
        start_time = time.time()
        result = pd.DataFrame()
        self.googlenews.set_lang(lang)
        self.googlenews.set_period('1d')
        self.googlenews.set_encode('utf-8')
        today = date.today()
        for keyword in self.keywords:
            self.googlenews.clear()
            self.googlenews.get_news(keyword)
            count = 0
            for row in  self.googlenews.results():
                row['keyword'] = keyword
                row['date'] = today
                result = result.append(row, ignore_index = True)
                count += 1
            logging.debug('found {} headlines for {} keyword'.format(count, keyword))
        
        if len(result) > 0:
            if 'datetime' in result.columns and 'img' in result.columns:
                return result.drop(columns=['datetime', 'img'])
        
        logging.info('Search complete: {}'.format((time.time() - start_time)))
        return result
                    
     
    def format_date(self, date):

        str_date = str(date.date())
        temp = str_date.strip().split('-')
        
        return "{}/{}/{}".format(temp[1],temp[-1],temp[0])
        
    def historic_seach(self, pages = 1, lang = 'pt', start_date = '', end_date = '', save_step = True, step = 55, filename = 'temp_result'):
        
        result = pd.DataFrame()        
        self.googlenews.clear()
        self.googlenews.set_lang(lang)
        self.googlenews.set_encode('utf-8')
        
        
        # format days between start and end
        current_date = pd.to_datetime(start_date, format='%m/%d/%Y')
        end = pd.to_datetime(end_date, format='%m/%d/%Y')

        # Start seach
        count = 0
        while current_date <= end:
            start_time = time.time()
            
            day = self.format_date(current_date)

            
            for keyword in self.keywords:
                #setup
                self.googlenews.clear()
                self.googlenews.set_lang(lang)
                self.googlenews.set_encode('utf-8')
                self.googlenews.set_time_range(day, day)
                
                # Get the first page
                self.googlenews.search(keyword)
                temp =  pd.DataFrame.from_dict(self.googlenews.results())
                temp['keyword'] = keyword
                result = pd.concat([result, temp])
                
                # Get multiple pages
                if pages > 1:
                    for x in range(2,pages + 1):
                        try:
                            temp =  pd.DataFrame.from_dict(self.googlenews.page_at(x))
                            temp['keyword'] = keyword
                            result = pd.concat([result, temp])
                        except:
                            break
                time.sleep(3)
                count += 1
            print('Search complete for day {} : {}'.format(day,(time.time() - start_time)), end = ' ')
            print(count)
            
            # wait
            if count > step:
                result.to_csv('{}.csv'.format(filename) ,index = None)
                print('wait')
                time.sleep(1200)
                count = 0
            # update current date
            current_date = current_date + timedelta(days = 1)
        return result    