import requests


class ExtractMedicalIndicators:
    def __init__(self):
        metamap_tags_file = '../SemanticTypes_2018AB.txt'
        tags_metamap = {elem.split('|')[0]: elem.split('|')[2].replace('\n', '') for elem in
                        open(self.metamap_tags_file, 'r').readlines()}

    def get_metamap_tags(self, sentence: str):
        # Doc for Metamap:
        ## shambakey1/metamap2018
        mm_feats = [0] * len(self.tags_metamap.keys())
        try:
            metamap_url = 'http://localhost:8080/form'
            data = {'input': sentence, 'args': '-N'}
            res = requests.post(metamap_url, data=data)
        except:
            raise Exception("Are you sure that your medical information extractor dcker is running?")
        for pos, tag in enumerate(self.tags_metamap.keys()):
            for found in res.text.split('\n')[2:]:
                try:
                    for ffound in found.split('[')[1].split(']')[0].split(','):
                        if ffound == tag:
                            mm_feats[pos] += 1
                except IndexError:
                    pass
        return mm_feats
