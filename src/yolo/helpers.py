from coremltools.proto import FeatureTypes_pb2 as ft


def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        return spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        return spec.neuralNetworkRegressor
    else:
        raise ValueError("MLModel does not have a neural network")


def set_type_as_float32(feature):
    if feature.type.HasField('multiArrayType'):
        feature.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32


def set_type_as_double(feature):
    if feature.type.HasField('multiArrayType'):
        feature.type.multiArrayType.dataType = ft.ArrayFeatureType.DOUBLE