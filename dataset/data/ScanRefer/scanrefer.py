import json
from collections import defaultdict


class ScanRefer:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.scene_id = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['unique_id']] = data['descriptions']
                    self.imgs[data['unique_id']] = data['image']
                    self.answers[data['unique_id']] = data['answer']
                    self.scene_id[data['unique_id']] = data['scene_id']
                    