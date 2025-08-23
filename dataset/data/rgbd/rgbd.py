import json
from collections import defaultdict


class RGBD:
    def __init__(self, annotation_file=None):
        self.anns, self.imgs, self.answers, self.types = defaultdict(list), dict(), dict(), dict()
        if not annotation_file == None:
            with open(annotation_file, 'r') as reader:
                datas = json.load(reader)
                for data in datas:
                    self.anns[data['id']] = data['captions']
                    self.imgs[data['id']] = data['image']
                    self.answers[data['id']] = data['answer']
                    self.types[data['id']] = data['type']
