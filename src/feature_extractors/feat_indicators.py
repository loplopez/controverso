from abc import ABC
from typing import Dict

import numpy as np


class ExtractDiscourseIndicators:
    """
    Indicator Features: in component and preceeding: stab2016parsing -> appendix table C1
    - Forward (therefore,...)
    - Backward ("because, in addition,...")
    - Thesis Indicators (in my opinion,...)
    - Rebuttal Indicators (although,...)
    - First person (I, me, myself, ...)"""

    indicators = {
        'forward_ind': ['As a result', 'As the consequence', 'Because', 'Clearly', 'Consequently',
                        'Considering this subject', 'Furthermore', 'Hence', 'leading to the consequence', 'so', 'So',
                        'taking account on this fact', 'That is the reason why', 'The reason is that', 'Therefore',
                        'therefore', 'This means that', 'This shows that', 'This will result', 'Thus', 'thus',
                        'Thus, it is clearly seen that', 'Thus, it is seen', 'Thus, the example shows'],
        'backward_ind': ['Additionally', 'As a matter of fact', 'because', 'Besides', 'due to', 'Finally',
                         'First of all',
                         'Firstly', 'for example', 'For example', 'For instance', 'for instance', 'Furthermore',
                         'has proved it', 'In addition', 'In addition to this', 'In the first place',
                         'is due to the fact that', 'It should also be noted', 'Moreover', 'On one hand',
                         'On the one hand',
                         'On the other hand', 'One of the main reasons', 'Secondly', 'Similarly', 'since', 'Since',
                         'So',
                         'The reason', 'To begin with', 'To offer an instance', 'What is more'],
        'thesis_ind': ['All in all', 'All things considered', 'As far as I am concerned', 'Based on some reasons',
                       'by analyzing both the views', 'considering both the previous fact', 'Finally',
                       'For the reasons mentioned above', 'From explanation above', 'From this point of view',
                       'I agree that', 'I agree with', 'I agree with the statement that', 'I believe', 'I believe that',
                       'I do not agree with this statement', 'I firmly believe that', 'I highly advocate that',
                       'I highly recommend', 'I strongly believe that', 'I think that', 'I think the view is',
                       'I totally agree', 'I totally agree to this opinion', 'I would have to argue that',
                       'I would reaffirm my position that', 'In conclusion', 'in conclusion', 'in my opinion',
                       'In my opinion', 'In my personal point of view', 'in my point of view', 'In my point of view',
                       'In summary', 'In the light of the facts outlined above', 'it can be said that',
                       'it is clear that',
                       'it seems to me that', 'my deep conviction', 'My sentiments', 'Overall', 'Personally',
                       'the above explanations and example shows that', 'This, however', 'To conclude',
                       'To my way of thinking', 'To sum up', 'Ultimately'],
        'rebuttal_ind': ['Admittedly', 'although', 'Although', 'besides these advantages', 'but', 'But', 'Even though',
                         'even though', 'However', 'Otherwise'],
        'first_person': ['I', 'me', 'myself', 'mine', 'my']
    }

    def indicator_features(self, text):
        indicator_features = []
        for indicator_list in self.indicators.values():
            indicators = [int(indicator.lower() in text.lower()) for indicator in indicator_list]
            # extra feature representing which group of indicators is present
            if 1 in indicators:
                indicator_features = indicator_features + [1]
            else:
                indicator_features = indicator_features + [0]
            indicator_features = indicator_features + indicators

        return np.array(indicator_features)
