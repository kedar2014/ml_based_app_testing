# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: classnames_id_label_.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='classnames_id_label_.proto',
  package='ml_code',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x1a\x63lassnames_id_label_.proto\x12\x07ml_code\" \n\x04Item\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\n\n\x02id\x18\x02 \x01(\x05\"(\n\x08\x43lassMap\x12\x1c\n\x05Items\x18\x01 \x03(\x0b\x32\r.ml_code.Item')
)




_ITEM = _descriptor.Descriptor(
  name='Item',
  full_name='ml_code.Item',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='ml_code.Item.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='ml_code.Item.id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=71,
)


_CLASSMAP = _descriptor.Descriptor(
  name='ClassMap',
  full_name='ml_code.ClassMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Items', full_name='ml_code.ClassMap.Items', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=73,
  serialized_end=113,
)

_CLASSMAP.fields_by_name['Items'].message_type = _ITEM
DESCRIPTOR.message_types_by_name['Item'] = _ITEM
DESCRIPTOR.message_types_by_name['ClassMap'] = _CLASSMAP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Item = _reflection.GeneratedProtocolMessageType('Item', (_message.Message,), dict(
  DESCRIPTOR = _ITEM,
  __module__ = 'classnames_id_label__pb2'
  # @@protoc_insertion_point(class_scope:ml_code.Item)
  ))
_sym_db.RegisterMessage(Item)

ClassMap = _reflection.GeneratedProtocolMessageType('ClassMap', (_message.Message,), dict(
  DESCRIPTOR = _CLASSMAP,
  __module__ = 'classnames_id_label__pb2'
  # @@protoc_insertion_point(class_scope:ml_code.ClassMap)
  ))
_sym_db.RegisterMessage(ClassMap)


# @@protoc_insertion_point(module_scope)
