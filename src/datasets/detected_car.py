import json
from collections import defaultdict


class DetectedCar:
    def __init__(self, image_path, make, model, price, date_of_first_registration, bbox):
        self.image_path = image_path
        self.make = make
        self.model = model
        self.price = price
        self.date_of_first_registration = date_of_first_registration
        self.bbox = bbox
        self.cluster_idx = None

    def __repr__(self):
        return f"DectectedCar({self.image_path}, {self.make}, {self.model}, {self.price}, {self.date_of_first_registration}, {self.bbox})"

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, DetectedCar):
            return self.image_path == self.image_path
        return NotImplemented

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(self.image_path)

    def to_json(self):
        return json.dumps({'image': self.image_path,
                           'make': self.make,
                           'model': self.model,
                           'price': self.price,
                           'date_of_first_registration': self.date_of_first_registration,
                           'bbox': [self.bbox['x'],
                                    self.bbox['y'],
                                    self.bbox['w'],
                                    self.bbox['h']],
                           'cluster_idx': self.cluster_idx})

    @staticmethod
    def from_data_frame(df):
        for index, row in df.iterrows():
            yield DetectedCar(
                image_path=row[0],
                make=row[1],
                model=row[2],
                price=row[3],
                date_of_first_registration=row[4],
                bbox={'x': row[5],
                      'y': row[6],
                      'w': row[7],
                      'h': row[8],
                      }
            )

    @staticmethod
    def to_price_clusters(detected_cars, labels):
        clusters = defaultdict(list)

        for detected_car, label in zip(detected_cars, labels):
            clusters[label].append(detected_car.price)

        return [cluster for cluster in clusters.values()]
