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


from typing import Optional, Union
from pydantic import Extra,  BaseModel, Field, StrictFloat, StrictInt, constr, validator

class DockerWorkerConfigV2DockerObjectLevel(BaseModel):
    """
    DockerWorkerConfigV2DockerObjectLevel
    """
    crop_dataset_name: Optional[constr(strict=True)] = Field(None, alias="cropDatasetName", description="Identical limitations than DatasetName however it can be empty")
    padding: Optional[Union[StrictFloat, StrictInt]] = None
    task_name: Optional[constr(strict=True)] = Field(None, alias="taskName", description="Since we sometimes stitch together SelectionInputTask+ActiveLearningScoreType, they need to follow the same specs of ActiveLearningScoreType. However, this can be an empty string due to internal logic. ")
    __properties = ["cropDatasetName", "padding", "taskName"]

    @validator('crop_dataset_name')
    def crop_dataset_name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-zA-Z0-9 _-]*$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9 _-]*$/")
        return value

    @validator('task_name')
    def task_name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-zA-Z0-9_+=,.@:\/-]*$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9_+=,.@:\/-]*$/")
        return value

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
    def from_json(cls, json_str: str) -> DockerWorkerConfigV2DockerObjectLevel:
        """Create an instance of DockerWorkerConfigV2DockerObjectLevel from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerWorkerConfigV2DockerObjectLevel:
        """Create an instance of DockerWorkerConfigV2DockerObjectLevel from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DockerWorkerConfigV2DockerObjectLevel.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerWorkerConfigV2DockerObjectLevel) in the input: " + str(obj))

        _obj = DockerWorkerConfigV2DockerObjectLevel.parse_obj({
            "crop_dataset_name": obj.get("cropDatasetName"),
            "padding": obj.get("padding"),
            "task_name": obj.get("taskName")
        })
        return _obj

