# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from __future__ import annotations
import pprint
import re  # noqa: F401
import json


from typing import Any, Dict, Optional
from pydantic import Extra,  BaseModel, Field, StrictStr
from lightly.openapi_generated.swagger_client.models.crop_data import CropData
from lightly.openapi_generated.swagger_client.models.sample_meta_data import SampleMetaData
from lightly.openapi_generated.swagger_client.models.video_frame_data import VideoFrameData

class SampleCreateRequest(BaseModel):
    """
    SampleCreateRequest
    """
    file_name: StrictStr = Field(..., alias="fileName")
    thumb_name: Optional[StrictStr] = Field(None, alias="thumbName")
    exif: Optional[Dict[str, Any]] = None
    meta_data: Optional[SampleMetaData] = Field(None, alias="metaData")
    custom_meta_data: Optional[Dict[str, Any]] = Field(None, alias="customMetaData")
    video_frame_data: Optional[VideoFrameData] = Field(None, alias="videoFrameData")
    crop_data: Optional[CropData] = Field(None, alias="cropData")
    __properties = ["fileName", "thumbName", "exif", "metaData", "customMetaData", "videoFrameData", "cropData"]

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> SampleCreateRequest:
        """Create an instance of SampleCreateRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of meta_data
        if self.meta_data:
            _dict['metaData' if by_alias else 'meta_data'] = self.meta_data.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of video_frame_data
        if self.video_frame_data:
            _dict['videoFrameData' if by_alias else 'video_frame_data'] = self.video_frame_data.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of crop_data
        if self.crop_data:
            _dict['cropData' if by_alias else 'crop_data'] = self.crop_data.to_dict(by_alias=by_alias)
        # set to None if custom_meta_data (nullable) is None
        # and __fields_set__ contains the field
        if self.custom_meta_data is None and "custom_meta_data" in self.__fields_set__:
            _dict['customMetaData' if by_alias else 'custom_meta_data'] = None

        # set to None if video_frame_data (nullable) is None
        # and __fields_set__ contains the field
        if self.video_frame_data is None and "video_frame_data" in self.__fields_set__:
            _dict['videoFrameData' if by_alias else 'video_frame_data'] = None

        # set to None if crop_data (nullable) is None
        # and __fields_set__ contains the field
        if self.crop_data is None and "crop_data" in self.__fields_set__:
            _dict['cropData' if by_alias else 'crop_data'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SampleCreateRequest:
        """Create an instance of SampleCreateRequest from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SampleCreateRequest.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SampleCreateRequest) in the input: " + str(obj))

        _obj = SampleCreateRequest.parse_obj({
            "file_name": obj.get("fileName"),
            "thumb_name": obj.get("thumbName"),
            "exif": obj.get("exif"),
            "meta_data": SampleMetaData.from_dict(obj.get("metaData")) if obj.get("metaData") is not None else None,
            "custom_meta_data": obj.get("customMetaData"),
            "video_frame_data": VideoFrameData.from_dict(obj.get("videoFrameData")) if obj.get("videoFrameData") is not None else None,
            "crop_data": CropData.from_dict(obj.get("cropData")) if obj.get("cropData") is not None else None
        })
        return _obj

