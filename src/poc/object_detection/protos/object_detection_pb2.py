# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/object_detection.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1dprotos/object_detection.proto\x12\x0fobjectdetection\"N\n\x10\x44\x65tectionRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x17\n\x0fimage_file_path\x18\x02 \x01(\t\x12\x12\n\nmodel_name\x18\x03 \x01(\t\"C\n\x11\x44\x65tectionResponse\x12.\n\ndetections\x18\x01 \x03(\x0b\x32\x1a.objectdetection.Detection\"c\n\tDetection\x12\r\n\x05label\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\t\n\x01x\x18\x03 \x01(\x02\x12\t\n\x01y\x18\x04 \x01(\x02\x12\r\n\x05width\x18\x05 \x01(\x02\x12\x0e\n\x06height\x18\x06 \x01(\x02\x32\x62\n\x0fObjectDetection\x12O\n\x06\x44\x65tect\x12!.objectdetection.DetectionRequest\x1a\".objectdetection.DetectionResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.object_detection_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_DETECTIONREQUEST']._serialized_start=50
  _globals['_DETECTIONREQUEST']._serialized_end=128
  _globals['_DETECTIONRESPONSE']._serialized_start=130
  _globals['_DETECTIONRESPONSE']._serialized_end=197
  _globals['_DETECTION']._serialized_start=199
  _globals['_DETECTION']._serialized_end=298
  _globals['_OBJECTDETECTION']._serialized_start=300
  _globals['_OBJECTDETECTION']._serialized_end=398
# @@protoc_insertion_point(module_scope)
